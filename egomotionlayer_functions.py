

from egomotionlayer_functions import *
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from config import *
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


def difference_of_gaussian(size, sigma1, sigma2):
    # Create a grid of (x, y) coordinates using PyTorch
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y)

    # Create two Gaussian kernels with different sigmas
    gaussian1 = (torch.exp(-(x**2 + y**2) / (2 * sigma1**2)))/(np.sqrt(2*np.pi)*sigma1)

    gaussian2 = (torch.exp(-(x**2 + y**2) / (2 * sigma2**2)))/(np.sqrt(2*np.pi)*sigma2)

    # Calculate Difference of Gaussian (DoG)
    dog = gaussian1 - gaussian2
    # normalise the kernel between -1 and 1
    # dog = (dog - dog.min()) / (dog.max() - dog.min())
    # dog = dog / torch.max(dog)
    # dog = dog * 2 - 1
    return dog

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

def plot_kernel(kernel,size):
    #plot kernel 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y)
    ax.plot_surface(x.numpy(), y.numpy(), kernel.numpy(), cmap='jet')
    plt.show()

def h5load_data(filename):
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]  # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        print(ds_arr)
    return data, ds_arr


class WinnerTakesAll(nn.Module):
    def __init__(self, k=1):
        super(WinnerTakesAll, self).__init__()
        self.k = k

    def forward(self, x):
        # Flatten the input except for the batch dimension
        flat_x = x.view(x.size(0), -1)
        # Get the top-k values and their indices
        topk_vals, topk_indices = torch.topk(flat_x, self.k, dim=1)
        # Create a mask of the same shape as flat_x
        mask = torch.zeros_like(flat_x)
        # Set the top-k values in the mask to 1
        mask.scatter_(1, topk_indices, 1)
        # Reshape the mask to the original input shape
        mask = mask.view_as(x)
        # Apply the mask to the input
        return x * mask

def net_def(filter, tau_mem, num_pyr):
    # define our single layer network and load the filters
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net = nn.Sequential(
        nn.Conv2d(1, num_pyr, (size_krn,size_krn), stride=1, bias=False),
        sl.LIF(tau_mem),
    )
    net[0].weight.data = filter.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net



def npy_data(filePathOrName, tsFLAG):
    recording = np.load(filePathOrName)
    if tsFLAG:
        recording[:, 3] *= 1e3  # convert time from seconds to milliseconds
    rec = rf.unstructured_to_structured(recording,
                                        dtype=np.dtype(
                                            [('x', np.int16), ('y', np.int16), ('p', bool), ('t', int)]))
    return rec

def load_krn(path):
    filters = []
    filter = np.load(path)
    filter = torch.tensor(np.stack(filter).astype(np.float32))
    filter = filter.unsqueeze(0)
    return filter


def load_eventsnpy(polarity, dur_video, FPS, filePathOrName,tsFLAG):
    rec = npy_data(filePathOrName,tsFLAG)
    # find out maximum x and y

    ### values here are in milliseonds the max ts should be the duration of the video 4 sec
    max_x = rec['x'].max().astype(int)
    max_y = rec['y'].max().astype(int)
    max_ts = rec['t'].max()
    # use single polarity
    rec['p'][rec['p'] == False] = True
    rec['p'] = polarity
    sensor_size = (max_x + 1, max_y + 1, 1)
    time_wnd_frames=rec['t'].max()/dur_video/FPS
    # print(f"sensor size is {sensor_size}")
    # We have to convert the raw events into frames so that we can feed those to our network
    # We use a library called tonic for that https://tonic.readthedocs.io/en/latest/ as well as torchvision
    transforms = torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_wnd_frames),
        torch.tensor,
    ])
    frames = transforms(rec)
    return frames,max_y, max_x,time_wnd_frames

def pyr_res(num_pyr,frames):
    res = []
    for pyr in range(1, num_pyr + 1):
        res.append((int((frames[0][0].shape[0]) / pyr), int((frames[0][0].shape[1]) / pyr)))
    return res

def running_stats(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    alpha: float = 0.1,
    eps: float = 1e-9,
):
    """
    Exponential running statistics.
    """

    diff = x - mean
    inc = alpha * diff
    mean += inc
    var = (1.0 - alpha) * (var + diff * inc)
    sd = torch.sqrt(var + eps)

    return (mean, var, sd)


def plot_runningmean(numframes,time_wnd_frames,meanegomap,respath,title):
    print(meanegomap)
    time = np.arange(0, (numframes-1)*time_wnd_frames* 1e-3, time_wnd_frames* 1e-3)
    plt.plot(time, meanegomap)
    plt.xlabel('time [ms]')# plotting by columns
    plt.ylabel('running mean - network activity')
    # plt.show()
    plt.savefig(respath+'/'+title, dpi=300)  # Save as PNG with 300 dpi
    return


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

def run(filter, frames, max_x, max_y,time_wnd_frames):

    # Define motion parameters
    tau_mem = time_wnd_frames* 10**-3 #tau_mem in milliseconds
    #Initialize the network with the loaded filter
    net = net_def(filter, tau_mem, num_pyr)
    cnt=0
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    res = pyr_res(num_pyr, frames)
    for frame in frames:
        print(str(cnt) + " frame out of " + str(frames.shape[0]))
        frame = frame.to(device, dtype=net[0].weight.dtype)
        egomap = net(frame)
        #resize egomap to the original size
        egomap = torch.nn.functional.interpolate(egomap.unsqueeze(0), size=(max_y+1, max_x+1), mode='bilinear', align_corners=False).squeeze(0)
        #frame, egomap between 0 and 255
        frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
        egomap = (egomap - egomap.min()) / (egomap.max() - egomap.min()) * 255
        #values under a threashold are set to 0
        egomap[egomap < threshold] = 0
        # create suppression map
        suppression = torch.zeros((1, max_y + 1, max_x + 1), device=device)
        #where egomap is over the threashold suppression max = frame
        suppression[egomap >= threshold] = frame[egomap >= threshold]

        # Show the egomap
        if show_egomap:
            # plot the frame and overlap the max point of the saliency map with a red dot
            plt.clf()
            #subplot ahowing the frame and the egomap
            plt.subplot(1, 3, 1)
            plt.imshow(frame.squeeze(0).cpu().detach().numpy(), cmap='gray')
            plt.colorbar(shrink=0.3)
            plt.title('Frame')

            plt.subplot(1, 3, 2)
            plt.imshow(egomap.squeeze(0).cpu().detach().numpy(), cmap='jet')
            plt.colorbar(shrink=0.3)
            plt.title('Egomap Map')

            #plot suppression map
            plt.subplot(1, 3, 3)
            plt.imshow(suppression.squeeze(0).cpu().detach().numpy(), cmap='gray')
            plt.colorbar(shrink=0.3)
            plt.title('Suppression Map')

            plt.draw()
            plt.pause(0.001)
        if save_res:
            # save the plot in a video
            plt.savefig(respath + 'egomaps/egomap' + str(cnt) + '.png')
        cnt+=1




def load_eventsh5(polarity, time_wnd_frames, ds_arr):
    # structured the data
    ds_arr[:, 0] -= int(ds_arr[0][0])  # starting from the first timestep
    rec_data = rf.unstructured_to_structured(ds_arr,
                                             dtype=np.dtype(
                                                 [('t', int), ('x', np.int16), ('y', np.int16), ('p', bool)]))

    # get the sensor size
    max_x = rec_data['x'].max().astype(int)
    max_y = rec_data['y'].max().astype(int)
    max_ts = rec_data['t'].max()
    # use single polarity
    rec_data['p'][rec_data['p'] == False] = True
    rec_data['p'] = polarity
    sensor_size = (max_x + 1, max_y + 1, 1)
    # print(f"sensor size is {sensor_size}")
    # convert events into frames
    transforms = torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_wnd_frames),
        torch.tensor,
    ])
    frames = transforms(rec_data)
    return frames, max_y, max_x

def createfld(respath, namefld):
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(respath+namefld):
        os.makedirs(respath+namefld)
        print('Folder created')
    else:
        print('Folder already exists')

