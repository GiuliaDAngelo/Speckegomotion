import os
import numpy as np

# Load the .npz file
OMSpath = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/OMS/'

# look at dirs whitin the path
dirs_oms = [d for d in os.listdir(OMSpath) if os.path.isdir(os.path.join(OMSpath, d))]
num_seq = len(dirs_oms)


for dir in dirs_oms:
    dirs_seq = [d for d in os.listdir(OMSpath+dir) if os.path.isdir(os.path.join(OMSpath+dir, d))]
    means = []
    stds = []
    for seq in dirs_seq:
        IoUs = np.load(OMSpath + dir + '/' + seq + 'IoUs.pkl', allow_pickle=True)
        IoUs = np.array(IoUs)
        if not np.isnan(np.mean(IoUs)):
            means.append(np.mean(IoUs))
            stds.append(np.std(IoUs))
    print('Mean IoU for sequence ' + dir + ' is ' + str(round(np.mean(means), 4)))
    print('+/-' + dir + ' is ' + str(round(np.mean(stds), 2)))