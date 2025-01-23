import matplotlib.pyplot as plt
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import iv
from sinabs import layers as sl


# Set the device to use for computation (MPS if available, otherwise CPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')



def plot_VMkernels(VMkernels,thetas):
    num_filters = len(VMkernels)
    num_rows = (num_filters + (len(VMkernels)//2) - 1) // (len(VMkernels)//2)  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, (len(VMkernels)//2), figsize=(15, 3 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (filter, theta) in enumerate(zip(VMkernels, thetas)):
        ax = axes[i]
        ax.imshow(filter.cpu().numpy(), cmap='jet')
        ax.set_title(f'Theta: {theta:.2f}')
        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



def create_von_mises(width, height, r0, theta0, filter_type='NORMAL', w=1.0, w2=1.0):
    # Create a grid of coordinates centered at (0, 0)
    x = np.arange(width) - width / 2
    y = -(np.arange(height) - height / 2)
    X, Y = np.meshgrid(x, y)

    # Adjust the grid based on the filter type
    if filter_type == 'CENTERED':
        X += r0 * np.cos(theta0)
        Y += r0 * np.sin(theta0)
    elif filter_type == 'OPPOSITE':
        X += 2 * r0 * np.cos(theta0)
        Y += 2 * r0 * np.sin(theta0)

    # Compute the radius and angle for each point in the grid
    r = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    # Compute the von Mises distribution
    von_mises = np.exp(r0 * w * np.cos(theta - theta0)) / iv(0, w2 * (r - r0))
    return von_mises


class AttentionModuleLevel(nn.Module):
    def __init__(self, VM_radius=10, VM_radius_group=15, num_ori=4, b_inh=1.5, g_inh=1.0, w_sum=0.5, vm_w=.1, vm_w2=.4,
                 vm_w_group=.2, vm_w2_group=.4, random_init=False, lif_tau=0.01):
        super(AttentionModuleLevel, self).__init__()
        # Initialize parameters for the attention module level
        self.VM_radius = VM_radius
        self.VM_radius_group = VM_radius_group
        self.num_ori = num_ori
        self.b_inh = b_inh
        self.g_inh = g_inh
        self.w_sum = w_sum
        self.vm_w = vm_w
        self.vm_w2 = vm_w2
        self.vm_w_group = vm_w_group
        self.vm_w2_group = vm_w2_group
        self.random_init = random_init
        self.lif_tau = lif_tau

        # Initialize convolutional layers and other variables
        self.initialize_conv_layers()

    def initialize_conv_layers(self):
        # Define orientations and initialize various lists for storing intermediate results
        self.orientations = np.arange(0, 180, 180 / self.num_ori)
        self.border_pyr_pos = []
        self.border_pyr_neg = []
        self.border_pyr_1_3 = []
        self.border_pyr_2_4 = []
        self.temp_border = []
        self.temp_border1 = []
        self.temp_border2 = []
        self.temp_border3 = []
        self.temp_border4 = []
        self.group_pyr1 = []
        self.group_pyr2 = []
        self.group_pyramid = []
        self.temp_pyramid = []
        self.vm_filters = []
        self.vm_filters_group = []

        # Calculate the sizes of the von Mises filters
        self.border_VM_size = int(self.VM_radius * 1.1)#2)
        self.group_VM_size = int(self.VM_radius_group * 1.5)#2.5)
        if self.border_VM_size % 2 == 0:
            self.border_VM_size += 1
        if self.group_VM_size % 2 == 0:
            self.group_VM_size += 1
        self.border_padding = (np.array(self.border_VM_size) - 1) // 2
        self.group_padding = (np.array(self.group_VM_size) - 1) // 2

        # Define convolutional layers for border and group processing
        self.border_conv = nn.Sequential(
            nn.Conv2d(1, self.num_ori * 2, self.border_VM_size, padding=self.border_padding, bias=False).to(device),
            sl.LIF(self.lif_tau)
        )
        channels = self.num_ori * 4
        self.group_conv = nn.Sequential(
            nn.Conv2d(channels, channels, self.group_VM_size, padding=self.group_padding, groups=channels,
                      bias=False).to(device),
            sl.LIF(self.lif_tau)
        )

        self.lif = sl.LIF(self.lif_tau).to(device)
        if not self.random_init:
            self.initialize_weights()
        self.initialized = False

    def initialize_weights(self):
        # Initialize weights for the convolutional layers using von Mises filters
        for ori in self.orientations:
            opp_ori = ori + 180
            self.vm_filters.append(torch.tensor(
                create_von_mises(self.border_VM_size, self.border_VM_size, self.VM_radius, ori * np.pi / 180,
                                 'CENTERED', self.vm_w, self.vm_w2)))
            self.vm_filters.append(torch.tensor(
                create_von_mises(self.border_VM_size, self.border_VM_size, self.VM_radius, opp_ori * np.pi / 180,
                                 'CENTERED', self.vm_w, self.vm_w2)))

            self.vm_filters_group += [torch.tensor(
                create_von_mises(self.group_VM_size, self.group_VM_size, self.VM_radius_group, ori * np.pi / 180,
                                 'NORMAL', self.vm_w, self.vm_w2))] * 2
            self.vm_filters_group += [torch.tensor(
                create_von_mises(self.group_VM_size, self.group_VM_size, self.VM_radius_group, opp_ori * np.pi / 180,
                                 'NORMAL', self.vm_w_group, self.vm_w2_group))] * 2
        for i in range(len(self.vm_filters)):
            self.vm_filters[i] = self.vm_filters[i] / (self.vm_filters[i].max() / 2) - 1
        self.vm_filters = torch.stack(self.vm_filters).unsqueeze(1).float().to(device)
        self.vm_filters_group = torch.stack(self.vm_filters_group).unsqueeze(1).float().to(device)
        with torch.no_grad():
            self.border_conv[0].weight.data = self.vm_filters.to(device)
            self.border_conv[1].v_mem = self.border_conv[1].tau_mem * self.border_conv[1].v_mem.to(device)

            self.group_conv[0].weight.data = self.vm_filters_group.to(device)
            self.group_conv[1].v_mem = self.group_conv[1].tau_mem * self.group_conv[1].v_mem.to(device)

    def compute_border_ownership(self, pyr):
        # Compute border ownership by applying convolutional layers to the input pyramid
        self.border_pyr_pos = self.border_conv(pyr[:, 0, ...].unsqueeze(1).to(device))
        self.border_pyr_neg = self.border_conv(pyr[:, 1, ...].unsqueeze(1).to(device))
        vMap = pyr.sum(1).to(device)
        for j in range(self.border_pyr_pos.shape[1] // 2):
            ori = 2 * j
            opp_ori = ori + 1
            border_pyr1 = self.lif(
                vMap * (self.border_pyr_pos[:, ori, ...] - self.border_pyr_neg[:, opp_ori, ...] * self.b_inh))
            border_pyr2 = self.lif(
                vMap * (self.border_pyr_pos[:, opp_ori, ...] - self.border_pyr_neg[:, ori, ...] * self.b_inh))
            border_pyr3 = self.lif(
                vMap * (self.border_pyr_neg[:, ori, ...] - self.border_pyr_pos[:, opp_ori, ...] * self.b_inh))
            border_pyr4 = self.lif(
                vMap * (self.border_pyr_neg[:, opp_ori, ...] - self.border_pyr_pos[:, ori, ...] * self.b_inh))

            self.border_pyr_1_3[j] = border_pyr1 + border_pyr3
            self.border_pyr_2_4[j] = border_pyr2 + border_pyr4

    def compute_grouping(self, border_pyr1_3, border_pyr2_4):
        # Compute grouping by combining border ownership results
        with torch.no_grad():
            for o, [b1, b2] in enumerate(zip(border_pyr1_3, border_pyr2_4)):
                self.temp_pyramid[o] = ((b1 - b2).abs())
                if o == 0:
                    self.temp_border = self.temp_pyramid[o]
                else:
                    self.temp_border = self.temp_border.max(self.temp_pyramid[o])

        for o, [b1, b2] in enumerate(zip(border_pyr1_3, border_pyr2_4)):
            wta_mask = torch.tensor(self.temp_pyramid[o] == self.temp_border, requires_grad=False).float().to(device)

            b1p = wta_mask * (b1 - b2)
            b1n = - b1p
            b1p = self.lif(b1p)
            b1n = self.lif(b1n)
            self.temp_border1[o] = (b1p * b1)
            self.temp_border2[o] = (b1p * b2 * self.g_inh)
            self.temp_border3[o] = (b1n * b2)
            self.temp_border4[o] = (b1n * b1 * self.g_inh)

        border_tensor = []
        for b in zip(self.temp_border1, self.temp_border2, self.temp_border3, self.temp_border4):
            border_tensor += b

        border_tensor = torch.stack(border_tensor)
        border_tensor.transpose_(1, 0)
        group_tensor = self.group_conv(border_tensor)

        for i in range(group_tensor.shape[1] // 4):
            self.group_pyr1[i] = self.lif(group_tensor[:, i * 4] - group_tensor[:, i * 4 + 1])
            self.group_pyr2[i] = self.lif(group_tensor[:, i * 4 + 2] - group_tensor[:, i * 4 + 3])

        self.group_pyramid = torch.stack(self.group_pyr1).sum(0) + torch.stack(self.group_pyr2).sum(0)

    def initialize_temp_variables(self, inp):
        # Initialize temporary variables for storing intermediate results
        batch_size = inp.shape[0]
        level_width = inp.shape[-2]
        level_height = inp.shape[-1]

        self.border_pyr_neg = torch.zeros((batch_size, 2 * self.num_ori, level_width, level_height), device=device)
        self.border_pyr_pos = torch.zeros((batch_size, 2 * self.num_ori, level_width, level_height), device=device)
        self.temp_border = torch.zeros((batch_size, level_width, level_height), device=device)

        self.group_pyramid = torch.zeros((batch_size, level_width, level_height), device=device)
        for i in range(self.num_ori):
            self.temp_border1.append(torch.zeros((batch_size, level_width, level_height), device=device))
            self.temp_border2.append(torch.zeros((batch_size, level_width, level_height), device=device))
            self.temp_border3.append(torch.zeros((batch_size, level_width, level_height), device=device))
            self.temp_border4.append(torch.zeros((batch_size, level_width, level_height), device=device))
            self.temp_pyramid.append(torch.zeros((batch_size, level_width, level_height), device=device))
            self.border_pyr_1_3.append(torch.zeros((batch_size, level_width, level_height), device=device))
            self.border_pyr_2_4.append(torch.zeros((batch_size, level_width, level_height), device=device))
            self.group_pyr1.append(torch.zeros((batch_size, level_width, level_height), device=device))
            self.group_pyr2.append(torch.zeros((batch_size, level_width, level_height), device=device))
        self.initialized = True

    def forward(self, inp):
        # Forward pass through the attention module level
        if not self.initialized:
            self.initialize_temp_variables(inp)

        self.compute_border_ownership(inp)
        self.compute_grouping(self.border_pyr_1_3, self.border_pyr_2_4)

        return self.group_pyramid


class AttentionModule(nn.Module):
    def __init__(self, VM_radius=10, VM_radius_group=15, num_ori=4, b_inh=1.5, g_inh=1.0, w_sum=0.5, vm_w=.1, vm_w2=.4,
                 vm_w_group=.2, vm_w2_group=.4, random_init=False, lif_tau=0.01):
        super(AttentionModule, self).__init__()
        # Initialize parameters for the attention module
        self.pyramid_levels = 1
        self.collapse_level = 0
        self.attention_levels = nn.ModuleList()
        self.group_pyramid = [None] * self.pyramid_levels
        for i in range(self.pyramid_levels):
            self.attention_levels.append(
                AttentionModuleLevel(VM_radius=VM_radius, VM_radius_group=VM_radius_group, num_ori=num_ori, b_inh=b_inh,
                         g_inh=g_inh, w_sum=w_sum, vm_w=vm_w, vm_w2=vm_w2, vm_w_group=vm_w_group,
                         vm_w2_group=vm_w2_group, random_init=random_init, lif_tau=lif_tau))


    @staticmethod
    def rescale_input(inp, xout_size, yout_size):
        # Rescale input to the desired size using grid sampling
        x = torch.linspace(-1, 1, yout_size, device=device).repeat(xout_size, 1)
        y = torch.linspace(-1, 1, xout_size, device=device).view(-1, 1).repeat(1, yout_size)
        grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
        grid = grid.unsqueeze_(0).repeat(inp.shape[0], 1, 1, 1)
        return F.grid_sample(inp, grid)

    def forward(self, inp: torch.Tensor):
        # Forward pass through the pyramid levels
        for l in range(0, self.pyramid_levels):
            level_width, level_height = (np.array(inp.shape[-2:]) * (0.7071 ** l)).astype(int)
            level = self.rescale_input(inp, level_width, level_height)
            self.group_pyramid[l] = self.attention_levels[l](level)

        # Combine the results from all pyramid levels to create the final saliency map
        saliency_map = self.rescale_input(self.group_pyramid[0].unsqueeze(0),
                                          self.group_pyramid[self.collapse_level].shape[-2],
                                          self.group_pyramid[self.collapse_level].shape[-1]).squeeze()

        for l in range(1, self.pyramid_levels):
            saliency_map += self.rescale_input(self.group_pyramid[l].unsqueeze(0),
                                               inp.shape[-2],
                                               inp.shape[-1]).squeeze()

        saliency_map /= saliency_map.max()
        return saliency_map
