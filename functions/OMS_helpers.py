import sinabs.layers as sl
import torch
import torch.nn as nn
import os



def OMSkernels(size_krn_center, sigma_center, size_krn_surround, sigma_surround):
    # create kernel Gaussian distribution
    center = gaussian_kernel(size_krn_center, sigma_center).unsqueeze(0)
    surround = gaussian_kernel(size_krn_surround, sigma_surround).unsqueeze(0)
    return center, surround

def gaussian_kernel(size, sigma):
    # Create a grid of (x, y) coordinates using PyTorch
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y, indexing='ij')  # Ensure proper indexing for 2D arrays
    # Create a Gaussian kernel
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # Normalize the kernel so that the values are between 0 and 1
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    return kernel

def net_def(filter, tau_mem, in_ch, out_ch, size_krn, device, stride):
    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, (size_krn,size_krn),  stride=stride, bias=False),
        sl.LIF(tau_mem),
    )
    net[0].weight.data = filter.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net


def egomotion(window, net_center, net_surround, device, max_y, max_x,threshold):
    window = window.unsqueeze(0).float().to(device)
    center = net_center(window)
    surround = net_surround(window)
    events = center - surround
    events = 1 - (events - events.min())/(events.max() - events.min())
    indexes = events >= threshold

    if indexes.any():
        OMS = torch.zeros_like(events)
        OMS[indexes] = 255
    else:
        OMS = torch.zeros_like(events)

    # center = (center - center.min()) / (center.max() - center.min())
    # surround = (surround - surround.min()) / (surround.max() - surround.min())
    # center = center * 255
    # surround = surround * 255
    # events = events * 255
    # fig, axs = plt.subplots(1, 4, figsize=(15, 10))
    # axs[0].cla()
    # axs[1].cla()
    # axs[2].cla()
    # axs[3].cla()
    # axs[0].imshow(center[0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
    # axs[1].imshow(surround[0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
    # axs[2].imshow(events[0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
    # axs[3].imshow(OMS[0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
    # plt.draw()
    # plt.pause(0.001)
    return OMS, indexes


def mkdirfold(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Folder created')
    else:
        print('Folder already exists')


def initialize_oms(device,OMS_PARAMS):
    """Initialize OMS kernels and networks."""
    center, surround = OMSkernels(
        OMS_PARAMS['size_krn_center'], OMS_PARAMS['sigma_center'],
        OMS_PARAMS['size_krn_surround'], OMS_PARAMS['sigma_surround']
    )
    net_center = net_def(center, OMS_PARAMS['tau_memOMS'], 1, 1,
                         OMS_PARAMS['size_krn_center'], device, OMS_PARAMS['sc'])
    net_surround = net_def(surround, OMS_PARAMS['tau_memOMS'], 1, 1,
                           OMS_PARAMS['size_krn_surround'], device, OMS_PARAMS['ss'])
    return net_center, net_surround