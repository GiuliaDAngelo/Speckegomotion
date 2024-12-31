import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rf
import torch
import torchvision
import tonic
from natsort import natsorted
import os
import cv2
from bimvee.importIitYarp import importIitYarp


def load_yarp_events(filePathOrName,codec,camera_events):
    # Load the event data using the 'importIitYarp' function
    # 'events' will contain the structured event data from the file
    events = importIitYarp(filePathOrName=filePathOrName,codec=codec)  # Codec to decode the event data
    # Extract the 'x' and 'y' coordinates of events, their timestamps ('ts'), and polarity ('pol')
    # These represent the x and y positions of the event, the time it occurred, and the polarity (whether it was a brightening or darkening event)
    e_x = events['data'][camera_events]['dvs']['x']  # x-coordinates of events
    e_y = events['data'][camera_events]['dvs']['y']  # y-coordinates of events
    e_ts = np.multiply(events['data'][camera_events]['dvs']['ts'], 10 ** 3)  # Convert timestamps to milliseconds
    e_pol = events['data'][camera_events]['dvs']['pol']  # Event polarity (1 for ON events, 0 for OFF events)
    return e_x,e_y,e_ts,e_pol

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

