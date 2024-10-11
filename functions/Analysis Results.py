import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
from natsort import natsorted

matplotlib.use('TkAgg')

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

def load_files(respath):
    with open(respath + '/meanegomap.pkl', 'rb') as f:
        meanegomap = pickle.load(f)
    with open(respath + '/frames.pkl', 'rb') as f:
        frames = pickle.load(f)
    with open(respath + '/time_wnd_frames.pkl', 'rb') as f:
        time_wnd_frames = pickle.load(f)
    with open(respath + '/numframes.pkl', 'rb') as f:
        numframes = pickle.load(f)
    return meanegomap,frames,time_wnd_frames,numframes

def plot_runningmean(numframes,time_wnd_frames,meanegomap,respath,title):
    print(meanegomap)
    time = np.arange(0, (numframes-1)*time_wnd_frames* 1e-3, time_wnd_frames* 1e-3)
    plt.plot(time, meanegomap)
    plt.xlabel('time [ms]')# plotting by columns
    plt.ylabel('running mean - network activity')
    # plt.show()
    plt.savefig(respath+'/'+title, dpi=300)  # Save as PNG with 300 dpi
    return


if __name__ == '__main__':
    #Analysis mean over event-frames
    respath = "results/resIEBCS"
    #load data
    [meanegomap,frames,time_wnd_frames,numframes] = load_files(respath)
    title = 'egomotion'

    #plot running mean
    plot_runningmean(numframes,time_wnd_frames,meanegomap,respath,title)

    #create videos from frames
    create_video_from_frames(respath+'/egomaps', respath+'/'+title+'.mp4', fps=30)
    # create_video_from_tensor(frames, respath+'/'+title+'edframes.mp4', fps=30)

