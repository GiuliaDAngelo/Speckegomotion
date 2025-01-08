import numpy as np
import torchvision
import tonic
import torch
from functions.attention_helpers import AttentionModule
import os
from functions.OMS_helpers import initialize_oms, egomotion
import cv2

import matplotlib
matplotlib.use('TkAgg')


##### check attention parameters and save results accordingly


class Config:
    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.80,
        'tau_memOMS': 0.02,
        'sc': 1,
        'ss': 1
    }
    # Attention Parameters
    ATTENTION_PARAMS = {
        'VM_radius': 8,  # (R0)
        'VM_radius_group': 15,
        'num_ori': 4,
        'b_inh': 3,  # (w)
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2,  # (rho)
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.3
    }
    SHOWIMGS = False
    maxBackgroundRatio = 2
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def loadeventsEVIMO(evpath, dir, npz, file):
    ###### ###### ###### ###### ###### ######
    ###### convert events into frames ######
    ###### ###### ###### ###### ###### ######
    # timestamp, x, y, p
    evdata = np.load(evpath + dir + npz + file, allow_pickle=True)
    events = evdata['events']
    ev = np.zeros(len(events), dtype=[('t', 'f8'), ('x', 'i2'), ('y', 'i2'), ('p', 'b')])
    ev['t'] = events[:, 0]
    ev['x'] = events[:, 1].astype(int)
    ev['y'] = events[:, 2].astype(int)
    ev['p'] = events[:, 3].astype(bool)
    max_y = ev['y'].max() + 1
    max_x = ev['x'].max() + 1

    # Split events by polarity
    pos_ev = ev[ev['p'] == 1]  # Positive polarity (ON events)
    neg_ev = ev[ev['p'] == 0]  # Negative polarity (OFF events)

    index = evdata['index']
    # K = evdata['K']
    # D = evdata['D']
    # depth = evdata['depth']
    mask = evdata['mask']
    meta = evdata['meta']
    GT = meta.item()['frames']

    discretization = evdata['discretization']
    sensor_size = (max_x, max_y, 1)
    time_wnd_frames = discretization

    # Convert dictionary to structured NumPy array
    transforms = torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_wnd_frames),
        torch.tensor,
    ])
    evframes_pos = transforms(pos_ev)
    evframes_neg = transforms(neg_ev)
    return evframes_pos, evframes_neg, max_y, max_x, mask, GT, time_wnd_frames



def mkdirfold(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Folder created')
    else:
        print('Folder already exists')




# Load the .npz file
evimofld = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/'
evpath = evimofld + 'EVIMOevents/'
maskpath = evimofld + 'EVIMOmasks/'
maskfrpath = evimofld + 'EVIMOmasksframes/'

# look at dirs whitin the path
dirs_events = [d for d in os.listdir(evpath) if os.path.isdir(os.path.join(evpath, d))]


config = Config()
for dir in dirs_events:
    npz = '/npz/'
    #look at files in the dir
    files = [f for f in os.listdir(evpath+dir+npz) if f.endswith('.npz')]
    files = sorted(files)
    for file in files:
        seq_name = file.split('.')[0]
        maskfile = seq_name +'_masks.npy'

        evmaskdata = np.load(maskpath + dir + '/' + maskfile, allow_pickle=True)
        [evframes_pos, evframes_neg, max_y, max_x, mask, GT, time_wnd_frames] = loadeventsEVIMO(evpath, dir, npz, file)
        evmaskdata = np.load(maskpath+dir+'/'+maskfile, allow_pickle=True)

        #results folder
        res_path = dir+'/'+seq_name+'/'
        print(res_path)
        maskframeres = maskfrpath + res_path

        mkdirfold(maskframeres)

        # Initialize OMS network
        net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)

        # Initialize Attention modules
        net_attention = AttentionModule(**config.ATTENTION_PARAMS)

        # Run network on the sub-dataset
        time = 0
        i = 0
        # fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        IOUs = []
        timestamps = [gt['ts'] for gt in GT]
        for evframe_pos, evframe_neg in zip(evframes_pos, evframes_neg):
            if time >= timestamps[i]:
                curr_frame_pos = (evframe_pos[0] != 0.00).clone().detach().to(torch.int)
                curr_frame_neg = (evframe_neg[0] != 0.00).clone().detach().to(torch.int)
                OMS_pos, indexes_pos = egomotion(curr_frame_pos, net_center, net_surround, config.DEVICE, max_y, max_x,config.OMS_PARAMS['threshold'])
                OMS_neg, indexes_neg = egomotion(curr_frame_neg, net_center, net_surround, config.DEVICE, max_y, max_x,config.OMS_PARAMS['threshold'])

                OMS_pos = OMS_pos.squeeze(0).squeeze(0).cpu().detach().numpy()
                OMS_neg = OMS_neg.squeeze(0).squeeze(0).cpu().detach().numpy()
                vSliceOMS = np.expand_dims(np.stack((OMS_pos, OMS_neg)), 0)
                with torch.no_grad():
                    saliency_mapOMS = net_attention(
                        torch.tensor(vSliceOMS, dtype=torch.float32).to(config.DEVICE)
                    ).cpu().numpy()

                cv2.imshow('Events map pos', curr_frame_pos.cpu().detach().numpy().astype(np.uint8)*255)
                cv2.imshow('Events map neg', curr_frame_neg.cpu().detach().numpy().astype(np.uint8)*255)
                OMS = OMS_pos + OMS_neg
                OMS[OMS != 0] = 1.0 * 255
                cv2.imshow('OMS map', OMS)
                cv2.imshow('Saliency map OMS', cv2.applyColorMap(
                    (saliency_mapOMS*255).astype(np.uint8), cv2.COLORMAP_JET))
                cv2.waitKey(1)
                i += 1
            time += time_wnd_frames


