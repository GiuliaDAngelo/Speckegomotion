'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script visualize the events from the DVS sensor.
'''
import numpy
import numpy as np
from scipy.special import iv
import cv2
import sinabs.backend.dynapcnn.io as sio
import samna
import time
import threading
import random
from configSpeckmain import *
import torch
import torch.nn as nn
import sinabs.layers as sl
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
import torchvision

def send_command(ser, command):
    ser.write(command.encode('utf-8'))
    time.sleep(0.1)  # Small delay to allow the device to process the command


def egokernel():
    # create kernel Gaussian distribution
    gauss_kernel_center = gaussian_kernel(size_krn_center, sigma_center)
    gauss_kernel_surround = gaussian_kernel(size_krn_surround, sigma_surround)

    # plot_kernel(gauss_kernel_center,gauss_kernel_center.size(0))
    # plot_kernel(gauss_kernel_surround,gauss_kernel_surround.size(0))

    filter = gauss_kernel_surround - gauss_kernel_center
    # plot_kernel(filter, filter.size(0))
    filter = filter.unsqueeze(0)
    return filter


def run_attention(window, net, device):
    # Create resized versions of the frames
    resized_frames = [torchvision.transforms.Resize((int(window.shape[2] / pyr), int(window.shape[1] / pyr)))(
        torch.from_numpy(window)) for pyr in range(1, num_pyr + 1)]
    # Process frames in batches
    batch_frames = torch.stack(
        [torchvision.transforms.Resize((resolution[0], resolution[1]))(frame) for frame in resized_frames]).type(torch.float32)
    batch_frames = batch_frames.to(device)  # Move to GPU if available
    output_rot = net(batch_frames)
    # Sum the outputs over rotations and scales
    salmap = torch.sum(torch.sum(output_rot, dim=1, keepdim=True), dim=0, keepdim=True).squeeze().type(torch.float32)
    salmax_coords = np.unravel_index(torch.argmax(salmap).cpu().numpy(), salmap.shape)
    # normalise salmap for visualization
    salmap = salmap.detach().cpu()
    salmap = np.array((salmap - salmap.min()) / (salmap.max() - salmap.min()) * 255)
    # rescale salmap to the original size
    # salmap = resize(salmap, (window.shape[1], window.shape[2]), anti_aliasing=False)
    return salmap,salmax_coords

def network_init(filters):
    """
    Initialize a neural network with a single convolutional layer using von Mises filters.

    Args:
        filters (torch.Tensor): Filters to be loaded into the convolutional layer.

    Returns:
        net (nn.Sequential): A simple neural network with one convolutional layer.
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Define a sequential network with a Conv2D layer followed by an IAF layer
    net = nn.Sequential(
        nn.Conv2d(1, filters.shape[1], filters.shape[2], stride=1, bias=False),
        # sl.IAF()
        sl.LIF(tau_mem=tau_mem),
        # #add winner take all layer
        # WinnerTakesAll(k=1)  # Add the WTA layer here
    )
    # Load the filters into the network weights
    net[0].weight.data = filters.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net

def plot_filters(filters, angles):
    """
    Plot the von Mises filters using matplotlib.

    Args:
        filters (torch.Tensor): A tensor containing filters to be visualized.
    """
    # Create subplots for 8 orientation VM filters
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle(f'VM filters size ({filters.shape[1]},{filters.shape[2]})', fontsize=16)

    # Display filters with their corresponding angles
    for i in range(8):
        if i < 4:
            axes[0, i].set_title(f"{round(angles[i],2)} grad")
            axes[0, i].imshow(filters[i])
            plt.colorbar(axes[0, i].imshow(filters[i]))
        else:
            axes[1, i - 4].set_title(f"{round(angles[i],2)} grad")
            axes[1, i - 4].imshow(filters[i])
            plt.colorbar(axes[1, i - 4].imshow(filters[i]))
    # add color bar to see the values of the filters
    plt.show()


def zero_2pi_tan(x, y):
    """
    Compute the angle in radians between the positive x-axis and the point (x, y),
    ensuring the angle is in the range [0, 2π].

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        angle (float): Angle in radians, between 0 and 2π.
    """
    angle = np.arctan2(y, x) % (2 * np.pi)  # Get the angle in radians and wrap it in the range [0, 2π]
    return angle

def vm_filter(theta, scale, rho=0.1, r0=0, thick=0.5, offset=(0, 0)):
    """Generate a Von Mises filter with r0 shifting and an offset."""
    height, width = scale, scale
    vm = np.empty((height, width))
    offset_x, offset_y = offset

    for x in range(width):
        for y in range(height):
            # Shift X and Y based on r0 and offset
            X = (x - width / 2) + r0 * np.cos(theta) - offset_x * np.cos(theta)
            Y = (height / 2 - y) + r0 * np.sin(theta) - offset_y * np.sin(theta)  # Inverted Y for correct orientation
            r = np.sqrt(X**2 + Y**2)
            angle = zero_2pi_tan(X, Y)

            # Compute the Von Mises filter value
            vm[y, x] = np.exp(thick*rho * r0 * np.cos(angle - theta)) / iv(0, r - r0)
    # normalise value between -1 and 1
    # vm = vm / np.max(vm)
    # vm = vm * 2 - 1
    return vm

def create_vm_filters(thetas, size, rho, r0, thick, offset):
    """
    Create a set of Von Mises filters with different orientations.

    Args:
        thetas (np.ndarray): Array of angles in radians.
        size (int): Size of the filter.
        rho (float): Scale coefficient to control arc length.
        r0 (int): Radius shift from the center.

    Returns:
        filters (list): List of Von Mises filters.
    """
    filters = []
    for theta in thetas:
        filter = vm_filter(theta, size, rho=rho, r0=r0, thick=thick, offset=offset)
        filter = rescale(filter, fltr_resize_perc, anti_aliasing=False)
        filters.append(filter)
    filters = torch.tensor(np.stack(filters).astype(np.float32))
    return filters


def plot_kernel(kernel,size):
    #plot kernel 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y, indexing='ij')
    ax.plot_surface(x.numpy(), y.numpy(), kernel.numpy(), cmap='jet')
    plt.show()


def fetch_events(sink, window, drop_rate, events_lock, numevs):
    while True:
        events = sink.get_events_blocking(1000) #ms
        if events:
            filtered_events = [event for event in events if random.random() > drop_rate]
            with events_lock:
                if filtered_events:
                    window[[event.y for event in filtered_events], [event.x for event in filtered_events]] = 255
                    numevs[0] += len(filtered_events)

def Specksetup():
    # List all connected devices
    device_map = sio.get_device_map()
    print(device_map)

    # Open the devkit device
    devkit = sio.open_device("speck2fdevkit:0")

    # Create and configure the event streaming graph
    samna_graph = samna.graph.EventFilterGraph()
    devkit_config = samna.speck2f.configuration.SpeckConfiguration()
    devkit_config.dvs_layer.raw_monitor_enable = True
    devkit.get_model().apply_configuration(devkit_config)
    sink = samna.graph.sink_from(devkit.get_model_source_node())
    samna_graph.start()
    devkit.get_stop_watch().start()
    devkit.get_stop_watch().reset()

    # Create an empty window for event visualization
    window = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
    numevs = [0]  # Use a list to allow modification within the thread
    events_lock = threading.Lock()
    return sink, window, numevs, events_lock

def net_def(filter, tau_mem, num_pyr, size_krn):
    # define our single layer network and load the filters
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net = nn.Sequential(
        nn.Conv2d(1, num_pyr, (size_krn,size_krn), stride=1, bias=False),
        sl.LIF(tau_mem),
    )
    net[0].weight.data = filter.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net

def gaussian_kernel(size, sigma):
    # Create a grid of (x, y) coordinates using PyTorch
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y, indexing='ij')

    # Create a Gaussian kernel
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    #kernel values between - 1 and 1
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    # kernel = kernel / torch.max(kernel)
    # kernel = kernel * 2 - 1
    return kernel

def egomotion(window, net, numevs, device):
    window = torch.from_numpy(window).unsqueeze(0).float().to(device)

    egomap = net(window)
    egomap = torch.nn.functional.interpolate(egomap.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
                                                   align_corners=False).squeeze(0)
    # frame, egomap between 0 and 255
    frame = (window - window.min()) / (window.max() - window.min()) * 255
    egomap = (egomap - egomap.min()) / (egomap.max() - egomap.min()) * 255
    suppression = torch.zeros((1, max_y, max_x), device=device)
    # where egomap is over the threashold suppression max = frame
    indexes = egomap >= threshold
    suppression[indexes] = frame[indexes]
    suppression=np.array(suppression.detach().cpu().numpy(), dtype=np.uint8)
    return suppression, indexes