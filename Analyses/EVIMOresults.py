import os
import numpy as np

# Load the .npz file
OMSpath = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/OMS/'

# Look at dirs within the path
dirs_oms = [d for d in os.listdir(OMSpath) if os.path.isdir(os.path.join(OMSpath, d))]
num_seq = len(dirs_oms)

for dir in dirs_oms:
    dirs_seq = [d for d in os.listdir(OMSpath + dir) if os.path.isdir(os.path.join(OMSpath + dir, d))]
    meansIoUs = []
    meansSSIMs = []
    for seq in dirs_seq:
        IoUs = np.load(OMSpath + dir + '/' + seq + 'meanIoUs.pkl', allow_pickle=True)
        IoUs = np.array(IoUs)

        SSIMs = np.load(OMSpath + dir + '/' + seq + 'meanSSIMs.pkl', allow_pickle=True)
        SSIMs = np.array(SSIMs)

        if IoUs.size > 0 and not np.isnan(IoUs).all():
            meansIoUs.append(np.nanmean(IoUs))
        if SSIMs.size > 0 and not np.isnan(SSIMs).all():
            meansSSIMs.append(np.nanmean(SSIMs))

    if meansIoUs:
        print('Mean IoU for sequence ' + dir + ' is ' + str(round(np.mean(meansIoUs)*100, 2)))
        print('+/-' + dir + ' is ' + str(round(np.std(meansIoUs), 2)))
    else:
        print('No valid IoU data for sequence ' + dir)

    if meansSSIMs:
        print('Mean SSIM for sequence ' + dir + ' is ' + str(round(np.mean(meansSSIMs), 2)))
        print('+/-' + dir + ' is ' + str(round(np.std(meansSSIMs), 2)))
    else:
        print('No valid SSIM data for sequence ' + dir)