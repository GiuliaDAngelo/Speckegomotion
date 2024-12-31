'''
Model developed by:
Giulia D'Angelo

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

from functions.OMS_helpers import *
from functions.data_helpers import *
import pickle
import matplotlib
matplotlib.use('TkAgg')

# Parameters kernel
size_krn_center = 8  # Size of the kernel (NxN) (all half ) - 8
sigma_center = 1  # Sigma for the first Gaussian - 1
size_krn_surround = 8  # Size of the kernel (NxN) - 8
sigma_surround = 4  # Sigma for the first Gaussian - 4

# Parameters network
threshold = 0.80
num_pyr = 1

# Parameters events
FPSvideo = 60.0  # Frames per second # Duration of video in seconds

# Flags
polarity = True
tsFLAG = False  # Flag to convert timestamps to microseconds
show_egomap = True
save_res = True
characterisationFLAG = True

# Paths
exp =  'objego'
# exp =  'ego'
# exp =  'onlyobj'

if characterisationFLAG:
    exp_kernel = 'krncenter' + str(size_krn_center) + 'sigcenter' + str(sigma_center) + 'krnsurround' + str(
        size_krn_surround) + 'sigsurround' + str(sigma_surround)
else:
    exp_kernel = ''

#Parameters stimuli
sf = 0.3  # spatial frequency [0.2, 1, 4]
sf_small = 3
speed = 0.01 # speed in cycles per frame
small_speed = 0.09
duration = 2  # duration in seconds


name_exp = exp+'sf'+str(sf)+'sp'+str(speed)+'sf'+str(sf_small)+'sp'+str(small_speed)
name_exp = name_exp.replace(".", "")

respath = 'results/'+name_exp+exp_kernel+'/'
evdata_path = 'ev_100_10_100_300_0.3_0.01.dat'
evpath = '/Users/giuliadangelo/workspace/code/IEBCS/data/video/stimuli/'+name_exp+'/'+name_exp+evdata_path+'.npy'


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

if __name__ == '__main__':

    # create results folders
    if save_res:
        create_results_folders(respath)

    # Load events
    time_wnd_frames = 1000 #1Khz as IEBCS
    [frames, max_y, max_x] = load_eventsnpy(polarity, duration, FPSvideo, evpath, tsFLAG, time_wnd_frames)
    len_fr = len(frames)
    time_wnd_frames = duration/len_fr
    tau_mem = time_wnd_frames

    #define egomotion network
    center, surround = OMSkernels(size_krn_center, sigma_center, size_krn_surround, sigma_surround)
    ss = 1
    sc = 1 #ss + sigma_surround - sigma_center
    net_center = net_def(center, tau_mem, 1, 1, size_krn_center, device, sc)
    net_surround = net_def(surround, tau_mem, 1, 1, size_krn_surround, device, ss)

    cnt=0
    spikes = [[] for _ in range((max_x+1)*(max_y+1))]
    time = 0
    for frame in frames:
        time = time+time_wnd_frames
        print(str(cnt) + " frame out of " + str(frames.shape[0]))
        OMS, indexes = egomotion(frame[0], net_center, net_surround, device, max_y, max_x, threshold)
        #coordinates where indexes True
        indtrue = np.argwhere(indexes[0].cpu() == True)
        for i in range(len(indtrue[0])):
            x = indtrue[1][i]
            y = indtrue[0][i]
            spikes[x+(y*max_x)].append(time)
        # Show the egomap
        if show_egomap:
            plt.clf()
            plt.imshow(OMS[0].cpu().numpy(), cmap='gray', vmin=0, vmax=255)
            plt.colorbar(shrink=0.3)
            # plt.title('OMS Map')
            plt.draw()
            plt.pause(0.001)
        if save_res:
            # save the plot in a video
            plt.savefig(respath + 'egomaps/egomap' + str(cnt) + '.png')
        cnt+=1

    with open(respath+'spikes.pkl', 'wb') as f:
        pickle.dump(spikes, f)

    # Save the results as videos
    if save_res:
        create_video_from_frames(respath+'egomaps/',  respath+respath.split('/')[1]+'egomap.mp4', fps=FPSvideo)

    print('end')
