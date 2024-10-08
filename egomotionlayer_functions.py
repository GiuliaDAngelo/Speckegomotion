
import h5py
import sinabs.layers as sl
import torch
import torch.nn as nn
import numpy as np
import tonic
import torchvision
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
import os
import cv2
from natsort import natsorted





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


def net_def(filters):
    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(1, filters.shape[0], filters.shape[1], bias=False),
        sl.IAF()
    )
    net[0].weight.data = filters.unsqueeze(1)
    return net



def npy_data(filePathOrName, tsFLAG):
    recording = np.load(filePathOrName)
    if tsFLAG:
        recording[:, 3] *= 1e6  # convert time from seconds to microseconds
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