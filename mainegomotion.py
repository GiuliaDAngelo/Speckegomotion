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

import matplotlib
matplotlib.use('TkAgg')


if __name__ == '__main__':

    # create results folders
    if save_res:
        create_results_folders(respath)

    # loading egomotion kernel
    filter_egomotion = egokernel()

    # Initialize the network with the loaded filter
    netegomotion = net_def(filter_egomotion,tau_mem, num_pyr, filter_egomotion.size(1))

    # Load event-based data from a specified .npy file.
    [frames, max_y, max_x, time_wnd_frames] = load_eventsnpy(polarity, dur_video, FPSvideo, filePathOrName, tsFLAG)

    # run(filter, frames, max_x, max_y,time_wnd_frames)
    # Define motion parameters
    cnt=0
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    spikes = [[[] for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    time = 0
    for frame in frames.numpy():
        time = time+time_wnd_frames
        print(str(cnt) + " frame out of " + str(frames.shape[0]))
        frame = (frame[0] > 0).astype(int)
        egomap, indexes = egomotion(frame, netegomotion, 1, device,(max_y+1), (max_x+1))
        #coordinates where indexes True
        indtrue = np.where(indexes[0].cpu() == True)
        for i in range(len(indtrue[0])):
            x = indtrue[1][i]
            y = indtrue[0][i]
            spikes[y][x].append(1)
        # coordinates where indexes False
        indfalse = np.where(indexes[0].cpu() == False)
        for i in range(len(indfalse[0])):
            x = indfalse[1][i]
            y = indfalse[0][i]
            spikes[y][x].append(0)
        # Show the egomap
        if show_egomap:
            # plot the frame and overlap the max point of the saliency map with a red dot
            plt.clf()

            #plot suppression map
            plt.imshow(egomap.squeeze(0), cmap='gray')
            plt.colorbar(shrink=0.3)
            # plt.title('Suppression Map')

            plt.draw()
            plt.pause(0.001)
        if save_res:
            # save the plot in a video
            plt.savefig(respath + 'egomaps/egomap' + str(cnt) + '.png')
        cnt+=1

    with open(name_exp+'.pkl', 'wb') as f:
        pickle.dump(spikes, f)

    # Save the results as videos
    if save_res:
        create_video_from_frames(respath+'egomaps/',  respath+respath.split('/')[1]+'egomap.mp4', fps=FPSvideo)

    print('end')
