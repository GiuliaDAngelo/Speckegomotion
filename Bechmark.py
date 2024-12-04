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

from egomotionlayer_functions import *

import matplotlib
matplotlib.use('TkAgg')


# Parameters kernel
size_krn_center = 8  # Size of the kernel (NxN) (all half ) - 8
sigma_center = 2  # Sigma for the first Gaussian - 1
size_krn_surround = 8  # Size of the kernel (NxN) - 8
sigma_surround = 4  # Sigma for the first Gaussian - 4


tau_mem = 0.001
threshold = 0.96
num_pyr = 1

# Parameters events
polarity = True
FPSvideo = 60.0  # Frames per second
dur_video = 4  # Duration of video in seconds
tsFLAG = False  # Flag to convert timestamps to microseconds
# Parameters stimuli
stimspeed = 60.0  # Speed of stimulus in pixels per second/ 1000 ms
show_egomap = True
save_res = True
title = respath.split('/')[1]




if __name__ == '__main__':

    # loading egomotion kernel, work on the kernel ---> here
    # filter_egomotion = egokernel()
    center, surround = OMS()
    ss = 1
    sc = ss + sigma_surround - sigma_center

    # Initialize the network with the loaded filter
    # define our single layer network and load the filters
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net_center = net_def(center, tau_mem, num_pyr, size_krn_center, device, sc)
    net_surround = net_def(surround, tau_mem, num_pyr, size_krn_surround, device, ss)
    time_wnd_frames = 400

    # Specify the directory to search
    folder_path = "/Users/giuliadangelo/workspace/data/DATASETs/VicBenchmark/benchmark/data/"
    folders = [f for f in os.listdir(folder_path) if f != '.DS_Store']

    for folder in folders:
        print('folder: '+folder)
        results_folder = '/Users/giuliadangelo/workspace/data/DATASETs/VicBenchmark/benchmark/results/'+folder+'/'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            print('Folder created')
        else:
            print('Folder already exists')
        # List to store file paths
        frames_files = []
        # Iterate through all files in the folder
        files_path = folder_path+folder+'/'
        for file_name in os.listdir(files_path):
            if "frames" in file_name.lower():  # Check if 'frames' is in the filename (case insensitive)
                file_path = os.path.join(files_path, file_name)
                frames_files.append(file_path)
            # Load the files
            loaded_data = []
            # sort the files
            frames_files = sorted(frames_files)
            # load npy file
            for file_path in frames_files:
                results_path = file_path.split('seq')[1]
                print(file_path)
                frames = np.load(file_path)
                # shape of the frames
                max_x = frames.shape[3]
                max_y = frames.shape[2]
                # Define motion parameters
                cnt=0
                device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
                spikes = [[] for _ in range((max_x+1)*(max_y+1))]
                time = 0
                OMS_s = []
                for frame in frames.astype(int):
                    print(str(cnt) + " frame out of " + str(frames.shape[0]))
                    OMS, indexes = egomotion(frame[0], net_center, net_surround,device, max_y, max_x, threshold)
                    #normalise OMS[0] between 0 and 255
                    OMS = (OMS[0] - OMS[0].min()) / (OMS[0].max() - OMS[0].min()) * 255
                    OMS_s.append(OMS)
                    #coordinates where indexes True
                    indtrue = np.argwhere(indexes[0].cpu() == True)
                    for i in range(len(indtrue[0])):
                        x = indtrue[1][i]
                        y = indtrue[0][i]
                        spikes[x+(y*max_x)].append(time)
                    # Show the egomap
                    time = time + time_wnd_frames
                    if show_egomap:
                        # plot the frame and overlap the max point of the saliency map with a red dot
                        plt.clf()
                        #plot suppression map
                        plt.imshow(OMS.squeeze(0).cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
                        plt.colorbar(shrink=0.3)
                        # plt.title('Suppression Map')
                        plt.draw()
                        plt.pause(0.001)
                    cnt+=1

            with open(name_exp+'.pkl', 'wb') as f:
                pickle.dump(spikes, f)
            with open('seq'+results_path.split('.npy')[0]+'.pkl', 'wb') as f:
                pickle.dump(OMS_s, f)



                print('end')
