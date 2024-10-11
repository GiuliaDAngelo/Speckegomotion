

experiments = 'objego'
# experiments = 'ego'
# experiments = 'onlyobj'

if experiments == 'ego':
    filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego1/ego1.npy"
    respath = "results/ego"
elif experiments == 'objego':
    filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/objego/objego.npy"
    respath = "results/objego"
elif experiments == 'onlyobj':
    filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/onlyobj/onlyobj.npy"
    respath = "results/onlyobj"


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
show_egomap = False
show_netactivity = False
num_pyr = 5  # number of pyramid levels


import torch.nn.functional as F






