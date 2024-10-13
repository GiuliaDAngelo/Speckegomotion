

respath = 'results/objego/'
# experiments = 'ego/'
# experiments = 'onlyobj/'

if respath == 'results/ego/':
    filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego1/ego1.npy"
elif respath == 'results/objego/':
    filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/objego/objego.npy"
elif respath == 'results/onlyobj/':
    filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/onlyobj/onlyobj.npy"


# Parameters kernel
size = 27  # Size of the kernel (NxN)
sigma1 = 0.04 * size  # Sigma for the first Gaussian
sigma2 = 2 * size  # Sigma for the second Gaussian
# Parameters events
polarity = True
FPSvideo = 60.0  # Frames per second
dur_video = 4  # Duration of video in seconds
tsFLAG = False  # Flag to convert timestamps to microseconds
# Parameters stimuli
stimspeed = 30.0  # Speed of stimulus in pixels per second/ 1000 ms
# Network parameters
# make show_egomap global variable across functions
show_egomap = True
save_res = True
num_pyr = 5  # number of pyramid levels
# Extract the title from the result path
title = respath.split('/')[1]






