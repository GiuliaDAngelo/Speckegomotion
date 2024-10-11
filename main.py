'''
Model developed by:
Giulia D'Angelo
Alexander Hadjiivanov

Summary:
This script loads a Difference of Gaussian (DoG) filter, initializes a spiking neural network, and processes event-based data from a video file.
The results are saved as different types of maps (egomaps, meanmaps, varmaps, etc.) and converted into videos.
The events can be loaded either from .npy or .h5 files, and the filter is applied across frames to extract motion information.

Steps:
1. Load the DoG filter.
2. Load event-based data from an .npy file.
3. Run the spiking neural network on the data.
4. Save the generated frames and create videos from the resulting maps.
'''

from egomotionlayer import *

if __name__ == '__main__':
    # Step 1: Load DoG filter
    path_data = "dog_kernel.npy"
    filter = load_krn(path_data)

    # Step 2: Initialize the network with the loaded filter
    net = net_def(filter)

    ###################################
    ########### LOAD EVENTS ###########
    ###################################

    # Load event-based data from a specified .npy file.
    # This part of the script allows loading events in different formats,
    # including from .npy files or .h5 files. We're focusing on .npy here.

    filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego_objmoving/ego_objmoving.npy"
    # Other options to load different datasets
    # filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/onlyego/egomotion1/egomotion1.npy"
    # filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/onlyobj/onlyobj.npy"

    # Define the result path
    respath = "results/objego"
    # respath = "results/onlyobj"
    # respath = "results/ego"

    # Extract the title from the result path
    title = respath.split('/')[1]

    # Load the event-based data (IEBCS format) using the parameters
    polarity = True
    FPSvideo = 60.0  # Frames per second
    dur_video = 4    # Duration of video in seconds
    tsFLAG = False   # Flag to convert timestamps to microseconds

    # Load the events into frames
    [frames, max_y, max_x, time_wnd_frames] = load_eventsnpy(polarity, dur_video, FPSvideo, filePathOrName, tsFLAG)

    # Define motion parameters
    stimspeed = 30.0  # Speed of stimulus in pixels per second
    deltaT = time_wnd_frames
    tau = time_wnd_frames * 4
    alpha = np.exp(-deltaT / tau)

    ###################################
    ########### RUN NETWORK ###########
    ###################################

    # Create folders for saving different types of maps (egomaps, meanmaps, etc.)
    createfld(respath, '/egomaps')
    createfld(respath, '/meanmaps')
    createfld(respath, '/varmaps')
    createfld(respath, '/sdmaps')
    createfld(respath, '/incmotionmaps')

    # Initialize the pyramid resolution
    num_pyr = 1
    res = pyr_res(num_pyr, frames)  # Get pyramid resolution
    egomaps = torch.empty((frames.shape[0], 1, max_y, max_x), dtype=torch.int64)

    # Get the total number of frames
    numframes = len(frames)

    # Skip the first frame
    frames = frames[1:numframes, :, :, :]

    # Run the network to generate egomaps
    egomaps = run(res, filter, egomaps, net, frames, max_x, max_y, num_pyr, respath, alpha)

    # Save the generated frames and metadata
    with open(respath + '/frames.pkl', 'wb') as f:
        pickle.dump(frames, f)
    with open(respath + '/time_wnd_frames.pkl', 'wb') as f:
        pickle.dump(time_wnd_frames, f)
    with open(respath + '/numframes.pkl', 'wb') as f:
        pickle.dump(numframes, f)

    ########### ANALYSIS RESULTS ###########

    # Convert the resulting frames into videos for each type of map
    create_video_from_frames(respath + '/egomaps', respath + '/' + title + '.mp4', fps=FPSvideo)
    create_video_from_frames(respath + '/meanmaps', respath + '/' + title + 'mean.mp4', fps=FPSvideo)
    create_video_from_frames(respath + '/varmaps', respath + '/' + title + 'var.mp4', fps=FPSvideo)
    create_video_from_frames(respath + '/sdmaps', respath + '/' + title + 'sd.mp4', fps=FPSvideo)
    create_video_from_frames(respath + '/incmotionmaps', respath + '/' + title + 'incmotion.mp4', fps=FPSvideo)

    print('end')
