

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
    return kernel

def plot_kernel(dog_kernel,size):
    #plot kernel 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y)
    ax.plot_surface(x.numpy(), y.numpy(), dog_kernel.numpy(), cmap='jet')
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

def net_def(filters, tau_mem):
    # define our single layer network and load the filters
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net = nn.Sequential(
        nn.Conv2d(1, filters.shape[0], filters.shape[1], bias=False),
        sl.LIF(tau_mem),
        # WinnerTakesAll(k=1)  # Add the WTA layer here
        # sl.IAF()
    )
    net[0].weight.data = filters.unsqueeze(1).to(device)
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
    # Create results folders
    createfld('', respath)
    createfld(respath, '/egomaps')
    createfld(respath, '/compmaps')



# def run(filter, frames, max_x, max_y,time_wnd_frames):
#     # Define motion parameters
#     deltaT = time_wnd_frames #microseconds
#     tau = time_wnd_frames * 4
#     alpha = np.exp(-deltaT / tau)
#     tau_mem = time_wnd_frames* 10**-3 #tau_mem in milliseconds
#     # Step 2: Initialize the network with the loaded filter
#     net = net_def(filter, tau_mem)
#
#     # Initialize the pyramid resolution
#     res = pyr_res(num_pyr, frames)  # Get pyramid resolution
#     egomaps = torch.empty((frames.shape[0], 1, max_y, max_x), dtype=torch.int64)
#     # Get the total number of frames
#     numframes = len(frames)
#     # Skip the first frame
#     frames = frames[1:numframes, :, :, :]
#
#
#     cnt=0
#     egomap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
#     save_egomap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
#     incmotionmap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
#
#     meanrunstats = torch.zeros((max_y, max_x))
#     varrunstats = torch.zeros((max_y, max_x))
#     sdrunstats = torch.zeros((max_y, max_x))
#
#     for frame in frames:
#         print(str(cnt) + "frame out of " + str(frames.shape[0]))
#         #scales pyramid
#         for pyr in range(1, num_pyr+1):
#             Hin = res[pyr-1][0]
#             Win = res[pyr-1][1]
#             Hw = int(Hin - filter.size()[1] + 1)
#             Ww = int(Win - filter.size()[2] + 1)
#             # print(f"pyramid scale {pyr}")
#             #get resolution and resize input for the pyramid
#             frm_rsz = torchvision.transforms.Resize((res[pyr-1][0], res[pyr-1][1]))(frame)
#             with torch.no_grad():
#                 output = net(frm_rsz.float())
#             # print(output.shape)
#             #normalising over the neurons on the layer
#             egomap[0, (pyr - 1)] = torchvision.transforms.Resize((max_y, max_x))(output) / (Hw * Ww)
#
#         # sum egomaps from the pyramid + running stats for running mean
#         egomap = torch.sum(egomap, dim=1, keepdim=True)
#         # show the egomap plt.draw()
#         #put flag to show the egomap
#         if show_egomap:
#             plt.imshow(egomap[0][0].cpu().detach().numpy(), cmap='jet')
#             plt.draw()
#             plt.pause(0.0001)
#
#         #running mean activity of the network
#         [meanrunstats, varrunstats,sdrunstats] = running_stats(egomap[0][0], meanrunstats, varrunstats,alpha)
#
#         # egomap - running mean activity in time
#         incmotionmap = egomap[0][0] - meanrunstats
#         if show_netactivity:
#             plt.imshow(incmotionmap.cpu().detach().numpy(), cmap='jet')
#             plt.draw()
#             plt.pause(0.0001)
#         #save results, normalise only for visualisation
#         egomaps[cnt] = egomap
#         save_egomap = (egomap - egomap.min()) / (egomap.max() - egomap.min())
#         incmotionmap  = (incmotionmap - incmotionmap.min()) / (incmotionmap.max() - incmotionmap.min())
#
#         #save egomap and meanmaps+varmaps+sdmaps
#         vutils.save_image(save_egomap.squeeze(0), respath+'/egomaps/egomap'+str(cnt)+'.png')
#         vutils.save_image(incmotionmap.squeeze(0), respath + '/incmotionmaps/incmotionmap' + str(cnt) + '.png')
#         plt.imsave(respath + '/meanmaps/meanmap'+str(cnt)+'.png', meanrunstats, cmap='jet')
#         plt.imsave(respath + '/varmaps/varmap'+str(cnt)+'.png', varrunstats, cmap='jet')
#         plt.imsave(respath + '/sdmaps/sdmap'+str(cnt)+'.png', sdrunstats, cmap='jet')
#
#         #empty egomaps
#         egomap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
#         save_egomap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
#         cnt+=1
#     return egomaps


def run(filter_dog, filter_gaus, frames, max_x, max_y,time_wnd_frames):

    # Define motion parameters
    tau_mem = time_wnd_frames* 10**-3 #tau_mem in milliseconds

    #Initialize the network with the loaded filter
    net_dog = net_def(filter_dog, tau_mem)
    net_gaus = net_def(filter_gaus, tau_mem)

    # Initialize the pyramid resolution
    res = pyr_res(num_pyr, frames)  # Get pyramid resolution

    cnt=0
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    for frame in frames:
        print(str(cnt) + " frame out of " + str(frames.shape[0]))
        # Create resized versions of the frames
        resized_frames = [torchvision.transforms.Resize((int(frame.shape[1] / pyr), int(frame.shape[2] / pyr)))(frame)
                          for pyr in range(1, num_pyr + 1)]
        # Process frames in batches
        batch_frames = torch.stack(
            [torchvision.transforms.Resize((max_y, max_x))(frame) for frame in resized_frames]).type(torch.float32)
        batch_frames = batch_frames.to(device)  # Move to GPU if available
        output = net_dog(batch_frames)
        mean_output = net_gaus(batch_frames)
        # Sum the outputs over rotations and scales
        egomap = torch.sum(output, dim=0, keepdim=True).squeeze().type(torch.float32)
        meanmap = torch.sum(mean_output, dim=0, keepdim=True).squeeze().type(torch.float32)
        # subtract to the egomap the mean activity of egomap
        # Subtract the mean activity of the network
        egomapsub = egomap - meanmap
        # # Normalise the egomap between 0 and 1
        egomap = (egomap - egomap.min()) / (egomap.max() - egomap.min())
        meanmap = (meanmap - meanmap.min()) / (meanmap.max() - meanmap.min())
        egomapsub = (egomapsub - egomapsub.min()) / (egomapsub.max() - egomapsub.min())
        # Show the egomap
        if show_egomap:
            # plot the frame and overlap the max point of the saliency map with a red dot
            plt.clf()
            # plot egomap
            plt.subplot(1, 3, 1)
            plt.imshow(egomap.cpu().detach().numpy(), cmap='jet')
            plt.title('Egomap')

            plt.subplot(1, 3, 2)
            plt.imshow(meanmap.cpu().detach().numpy(), cmap='jet')
            plt.title('Mean Map')

            plt.subplot(1, 3, 3)
            plt.imshow(egomapsub.cpu().detach().numpy(), cmap='jet')
            plt.title('Subtraction Map')

            # plot the mean activity of the network
            plt.draw()
            plt.pause(0.001)
        if save_res:
            # save the plot in a video
            plt.savefig(respath + 'compmaps/compmaps' + str(cnt) + '.png')
            # save egomap as image
            plt.imsave(respath + 'egomaps/egomap' + str(cnt) + '.png', egomap.cpu().detach(), cmap='jet')
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