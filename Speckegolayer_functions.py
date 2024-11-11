'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script visualize the events from the DVS sensor.
'''


import numpy as np
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

def plot_kernel(kernel,size):
    #plot kernel 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y)
    ax.plot_surface(x.numpy(), y.numpy(), kernel.numpy(), cmap='jet')
    plt.show()


def fetch_events(sink, window, drop_rate, events_lock, numevs):
    while True:
        events = sink.get_events_blocking(1000)
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
    x, y = torch.meshgrid(x, y)

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
    # center_map = net_center(window)
    # surround_map = net_surround(window)

    # resize egomap to the original size
    # center_map = torch.nn.functional.interpolate(center_map.unsqueeze(0), size=(max_y, max_x), mode='bilinear', align_corners=False).squeeze(0)
    # surround_map = torch.nn.functional.interpolate(surround_map.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
    #                                 align_corners=False).squeeze(0)

    egomap = torch.nn.functional.interpolate(egomap.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
                                                   align_corners=False).squeeze(0)

    # frame, egomap between 0 and 255
    frame = (window - window.min()) / (window.max() - window.min()) * 255
    # center_map = (center_map - center_map.min()) / (center_map.max() - center_map.min()) * 255
    # surround_map = (surround_map - surround_map.min()) / (surround_map.max() - surround_map.min()) * 255
    egomap = (egomap - egomap.min()) / (egomap.max() - egomap.min()) * 255
    # suppression = center_map - surround_map

    # values under a threashold are set to 0
    # center_map[center_map < threshold] = 0
    # create suppression map
    suppression = torch.zeros((1, max_y, max_x), device=device)
    # where egomap is over the threashold suppression max = frame
    indexes = egomap >= threshold
    suppression[indexes] = frame[indexes]
    # how many indexes are true
    return suppression, indexes