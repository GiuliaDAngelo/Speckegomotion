

from egomotionlayer_functions import *
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import pickle
import numpy as np
import h5py
import sinabs
import numpy.lib.recfunctions as rf
import sinabs.layers as sl
import torch
import torch.nn as nn
import torchvision
import tonic
from natsort import natsorted
import os
import cv2


def create_video_from_frames(frame_folder, output_video, fps=30):
    # Get all the frame file names and sort them
    frames = [img for img in os.listdir(frame_folder) if img.endswith(".png")]
    # frames.sort()  # Ensure the frames are in order
    frames = natsorted(frames)

    # Get the width and height of the first frame
    first_frame_path = os.path.join(frame_folder, frames[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_name in frames:
        frame_path = os.path.join(frame_folder, frame_name)
        frame = cv2.imread(frame_path)
        video.write(frame)  # Write the frame to the video

    video.release()  # Close the video writer
    cv2.destroyAllWindows()


def create_results_folders(respath):
    #check if teh folders exist already
    createfld('', respath)
    createfld(respath, '/egomaps')

def load_eventsnpy(polarity, dur_video, FPS, filePathOrName,tsFLAG,time_wnd_frames):
    rec = npy_data(filePathOrName,tsFLAG)
    # find out maximum x and y

    ### values here are in milliseonds the max ts should be the duration of the video 4 sec
    max_x = rec['x'].max().astype(int) + 1
    max_y = rec['y'].max().astype(int) + 1
    max_ts = rec['t'].max()
    # use single polarity
    rec['p'][rec['p'] == False] = True
    rec['p'] = polarity
    sensor_size = (max_x, max_y, 1)
    # print(f"sensor size is {sensor_size}")
    # We have to convert the raw events into frames so that we can feed those to our network
    # We use a library called tonic for that https://tonic.readthedocs.io/en/latest/ as well as torchvision
    transforms = torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_wnd_frames),
        torch.tensor,
    ])
    frames = transforms(rec)
    return frames,max_y, max_x


def egomotion(window, net_center, net_surround, device, max_y, max_x,threshold):
    window = torch.from_numpy(window).unsqueeze(0).float().to(device)
    center = net_center(window)
    surround = net_surround(window)
    center = torch.nn.functional.interpolate(center.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
                                             align_corners=False).squeeze(0)
    surround = torch.nn.functional.interpolate(surround.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
                                             align_corners=False).squeeze(0)
    events = center - surround
    events = (events - events.min()) / (events.max() - events.min())
    indexes = events > threshold
    if indexes.any():
        OMS = torch.zeros_like(events)
        OMS[indexes] = 255
    else:
        OMS = torch.zeros_like(events)
    return OMS, indexes


def npy_data(filePathOrName, tsFLAG):
    recording = np.load(filePathOrName)
    if tsFLAG:
        recording[:, 3] *= 1e3  # convert time from seconds to milliseconds
    rec = rf.unstructured_to_structured(recording,
                                        dtype=np.dtype(
                                            [('x', np.int16), ('y', np.int16), ('p', bool), ('t', int)]))
    return rec


def createfld(respath, namefld):
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(respath+namefld):
        os.makedirs(respath+namefld)
        print('Folder created')
    else:
        print('Folder already exists')



def savekernel(kernel, size, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = torch.linspace(0, size, size)
    y = torch.linspace(0, size, size)
    x, y = torch.meshgrid(x, y)
    ax.plot_surface(x.numpy(), y.numpy(), kernel[0].numpy(), cmap='jet')
    plt.show()
    plt.savefig(name+'.png')


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

def net_def(filter, tau_mem, num_pyr, size_krn, device, stride):
    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(1, num_pyr, (size_krn,size_krn), stride=stride, bias=False),
        sl.LIF(tau_mem),
    )
    net[0].weight.data = filter.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net


def egomotion(window, net_center, net_surround, device, max_y, max_x,threshold):
    window = window.unsqueeze(0).float().to(device)
    center = net_center(window)
    surround = net_surround(window)
    center = torch.nn.functional.interpolate(center.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
                                             align_corners=False).squeeze(0)
    surround = torch.nn.functional.interpolate(surround.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
                                             align_corners=False).squeeze(0)

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