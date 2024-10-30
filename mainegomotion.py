'''
Model developed by:
Giulia D'Angelo
Alexander Hadjiivanov

Summary:
This script loads a Gaussian filter, initializes a spiking neural network, and processes event-based data from a video file.
The results are saved as different types of maps (egomaps, meanmaps, varmaps, etc.) and converted into videos.
The events can be loaded either from .npy or .h5 files, and the filter is applied across frames to extract motion information.

Steps:
1. Load the Gaussian filter.
2. Load event-based data from an .npy file.
3. Run the spiking neural network on the data.
4. Save the generated frames and create videos from the resulting maps.
'''
from configmain import *
from egomotionlayer_functions import *


if __name__ == '__main__':

    # create results folders
    if save_res:
        create_results_folders(respath)

    # create kernel Gaussian distribution
    gauss_kernel = gaussian_kernel(size_krn, sigma)
    # plot_kernel(gauss_kernel,gauss_kernel.size(0))
    filter = gauss_kernel.unsqueeze(0)

    # Load event-based data from a specified .npy file.
    [frames, max_y, max_x, time_wnd_frames] = load_eventsnpy(polarity, dur_video, FPSvideo, filePathOrName, tsFLAG)

    # run(filter_dog, filter_gauss, frames, max_x, max_y,time_wnd_frames)
    run(filter, frames, max_x, max_y,time_wnd_frames)

    # Save the results as videos
    if save_res:
        create_video_from_frames(respath+'egomaps/',  respath+respath.split('/')[1]+'egomap.mp4', fps=FPSvideo)

    print('end')
