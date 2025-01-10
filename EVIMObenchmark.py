import numpy as np
import matplotlib.pyplot as plt
import os
from functions.OMS_helpers import initialize_oms, egomotion
import pickle
import torch.nn.functional as F
import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim

import matplotlib
matplotlib.use('TkAgg')

class Config:
    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.86,
        'tau_memOMS': 0.02,
        'sc': 1,
        'ss': 1
    }
    SHOWIMGS = False
    maxBackgroundRatio = 2
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')



def mkdirfold(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Folder created')
    else:
        print('Folder already exists')

def getIOU(spike_pred, spike_gt):
    spike_pred = spike_pred.cpu().numpy()
    spike_gt = spike_gt.cpu().numpy()

    intersection = np.sum(np.logical_and(spike_pred, spike_gt))
    union = np.sum(np.logical_or(spike_pred, spike_gt))
    return intersection/union



# Load the .npz file
evimofld = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/'
evpath = evimofld + 'EVIMOevents/'
maskpath = evimofld + 'EVIMOmasks/'
evfrpath = evimofld + 'EVIMOevframes/'
maskfrpath = evimofld + 'EVIMOmasksframes/'
OMSpath = evimofld + 'OMS/'

# look at dirs whitin the path
dirs_events = [d for d in os.listdir(evpath) if os.path.isdir(os.path.join(evpath, d))]
dirs = [d for d in os.listdir(evfrpath) if os.path.isdir(os.path.join(evfrpath, d))]


config = Config()
for dir in dirs:
    # look at files in the dir
    files = [f for f in os.listdir(evfrpath + dir) if f.endswith('.npy')]
    # sort list of files
    files = sorted(files)
    for file in files:
        seq_name = file.split('_')[0]+'_'+file.split('_')[1]
        maskfile = seq_name + '_masks.npy'
        res_path = dir + '/' + seq_name + '/'

        print(res_path)
        evframeres = evfrpath + res_path
        maskframeres = maskfrpath + res_path
        OMSframeres = OMSpath + res_path

        mkdirfold(evframeres)
        mkdirfold(maskframeres)
        mkdirfold(OMSframeres)

        evframesdata = np.load(evfrpath + dir + '/' + file, allow_pickle=True)
        evmaskdata = np.load(maskpath + dir + '/' + maskfile, allow_pickle=True)

        # results folder
        res_path = dir + '/' + seq_name + '/'
        print(res_path)
        OMSframeres = OMSpath + res_path
        mkdirfold(OMSframeres)

        # Initialize OMS network
        net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)

        i = 0
        if config.SHOWIMGS:
            fig, axs = plt.subplots(1, 4, figsize=(10, 5))
        IOUs = []
        SSIMs = []
        max_x = evframesdata[0].shape[2]
        max_y = evframesdata[0].shape[1]
        for evframe in evframesdata:
            # print(str(i)+' out of: '+str(len(evframesdata)))
            OMS, indexes = egomotion(torch.tensor(evframe[0]), net_center, net_surround, config.DEVICE, max_y, max_x, config.OMS_PARAMS['threshold'])

            # IoU
            dens_mask = torch.tensor(evmaskdata[i] != 0.00, dtype=torch.bool)
            spk_evframe = torch.tensor(evframe[0] != 0.00, dtype=torch.bool)
            spk_oms = F.interpolate(OMS[0].unsqueeze(0).unsqueeze(0), size=dens_mask.shape, mode='bilinear',
                                    align_corners=False).squeeze(0).squeeze(0).to(config.DEVICE).bool()

            spike_pred = torch.zeros_like(OMS[0]).bool()
            torch.logical_and(spk_oms, spk_evframe.to(config.DEVICE), out=spike_pred)

            spk_mask = torch.zeros_like(dens_mask).to(config.DEVICE)
            torch.logical_and(dens_mask.to(config.DEVICE),spk_evframe.to(config.DEVICE), out=spk_mask)

            spike_gt = torch.zeros_like(spike_pred)
            torch.logical_and(spike_pred.to(config.DEVICE), dens_mask.to(config.DEVICE), out=spike_gt)

            num_evs_mask = torch.sum(spk_mask).item()
            num_evs_back = torch.sum(spk_evframe).item() - num_evs_mask
            try:
                ratio = num_evs_back / num_evs_mask
            except ZeroDivisionError:
                ratio = float('inf')  # or any other value that makes sense in your context
            # print('ratio: '+ str(ratio))
            # print(num_evs_back / num_evs_mask)
            if ratio < config.maxBackgroundRatio:
                #IoU
                IOUframe = getIOU(spike_pred, spike_gt)
                IOUs.append(IOUframe)
                # SSIM
                spike_pred_ssim = spike_pred.unsqueeze(0).unsqueeze(0).float()
                spike_gt_ssim = spike_gt.unsqueeze(0).unsqueeze(0).float()
                ssimframe = ssim(spike_pred_ssim.float(), spike_gt_ssim.float(), data_range=1.0).item()
                SSIMs.append(ssimframe)
                print('IoU: ' + str(IOUframe))
                print('SSIM: ' + str(ssimframe))

                # Plot
                if config.SHOWIMGS:
                    axs[0].cla()
                    axs[1].cla()
                    axs[2].cla()

                indexes_gt = evmaskdata[i] != 0
                ground_truth = np.zeros_like(evmaskdata[i], dtype=evframe[0].dtype)
                ground_truth[indexes_gt] = evframe[0][indexes_gt]

                if config.SHOWIMGS:
                    axs[0].imshow(evframe[0])
                    axs[1].imshow(ground_truth)
                    axs[2].imshow(spike_gt.cpu().numpy())

                plt.imsave(evframeres + f'evframe_{i}.png', evframe[0], cmap='gray')
                plt.imsave(maskframeres + f'mask_{i}.png', spike_gt.cpu().numpy(),cmap='gray')
                plt.imsave(OMSframeres + f'OMS_{i}.png', spike_pred.cpu().numpy(), cmap='gray')
            if config.SHOWIMGS:
                plt.draw()
                plt.pause(0.001)
            i += 1

        with open(OMSpath + dir + '/' + seq_name + 'meanIOUs.pkl', 'wb') as f:
            pickle.dump(np.mean(IOUs), f)
        print('mean IOU for '+dir+' '+file+': ' + str(np.mean(IOUs)))

        with open(OMSpath + dir + '/' + seq_name + 'meanSSIMs.pkl', 'wb') as f:
            pickle.dump(np.mean(SSIMs), f)
        print('mean SSIM for ' + dir + file + ': ' + str(np.mean(SSIMs)))


print('end')
