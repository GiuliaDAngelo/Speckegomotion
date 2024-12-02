
# respath = 'results/objego/'
respath = 'results/ego/'
# respath = 'results/onlyobj/'

name_exp = respath.split('/')[1]

respath_to_file = {
    'results/ego/': "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego1/ego1.npy",
    'results/ego3/': "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego3/ego3.npy",
    'results/ego4/': "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego4/ego4.npy",
    'results/ego5/': "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego5/ego5.npy",
    'results/ego8/': "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego8/ego8.npy",
    'results/objego/': "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/objego/objego.npy",
    'results/onlyobj/': "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/onlyobj/onlyobj.npy"
}

filePathOrName = respath_to_file.get(respath, None)

# Parameters kernel
size_krn_center = 8  # Size of the kernel (NxN)
sigma_center = 1  # Sigma for the first Gaussian
size_krn_surround = 8  # Size of the kernel (NxN)
sigma_surround = 4  # Sigma for the first Gaussian


tau_mem = 0.01
threshold = 200
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









