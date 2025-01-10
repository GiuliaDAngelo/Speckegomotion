import numpy as np
import torchvision
import tonic
import torch
from functions.attention_helpers import AttentionModule
import os
from functions.OMS_helpers import initialize_oms, egomotion
import cv2
import pickle
import matplotlib.pyplot as plt

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
resultspath = evimofld + 'ATT/'

# look at dirs whitin the path
dirs_events = [d for d in os.listdir(evpath) if os.path.isdir(os.path.join(evpath, d))]


config = Config()
for dir in dirs_events:
    # if dir == "wall" or dir == "box" or dir == "fast" or dir == "table" or dir == "tabletop":
    #     continue
    npz = '/npz/'
    #look at files in the dir
    files = [f for f in os.listdir(evpath+dir+npz) if f.endswith('.npz')]
    files = sorted(files)
    for file in files:
        accuracy = []
        seq_name = file.split('.')[0]
        maskfile = seq_name +'_masks.npy'

        evmaskdata = np.load(maskpath + dir + '/' + maskfile, allow_pickle=True)
        [evframes_pos, evframes_neg, max_y, max_x, masks, GT, time_wnd_frames] = loadeventsEVIMO(evpath, dir, npz, file)

        #results folder
        seq_path = dir+'/'+seq_name+'/'
        print(seq_path)
        res_path = resultspath +seq_path
        mkdirfold(res_path+'evframes/')
        mkdirfold(res_path+'OMS/')
        mkdirfold(res_path+'salmaps/')

        # Initialize OMS network
        net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)

        # Initialize Attention modules
        net_attention = AttentionModule(**config.ATTENTION_PARAMS)

        # Run network on the sub-dataset
        time = 0
        i = 0
        max_to_object = 0
        cnt = 0
        timestamps = [gt['ts'] for gt in GT]
        for evframe_pos, evframe_neg in zip(evframes_pos, evframes_neg):
            if time >= timestamps[i]:
                #split in polarities
                curr_frame_pos = (evframe_pos[0] != 0.00).clone().detach().to(torch.int)
                curr_frame_neg = (evframe_neg[0] != 0.00).clone().detach().to(torch.int)

                # Load mask data
                mask = evmaskdata[i]

                # Compute OMS for polarities
                OMS_pos, indexes_pos = egomotion(curr_frame_pos, net_center, net_surround, config.DEVICE, max_y, max_x,config.OMS_PARAMS['threshold'])
                OMS_neg, indexes_neg = egomotion(curr_frame_neg, net_center, net_surround, config.DEVICE, max_y, max_x,config.OMS_PARAMS['threshold'])
                OMS_pos = OMS_pos.squeeze(0).squeeze(0).cpu().detach().numpy()
                OMS_neg = OMS_neg.squeeze(0).squeeze(0).cpu().detach().numpy()

                # Compute saliency map for OMS
                vSliceOMS = np.expand_dims(np.stack((OMS_pos, OMS_neg)), 0)
                with torch.no_grad():
                    saliency_mapOMS = (net_attention(
                        torch.tensor(vSliceOMS, dtype=torch.float32).to(config.DEVICE)
                    ).cpu().numpy())*255

                # ratio background to object events
                dens_mask = torch.tensor(mask != 0.00, dtype=torch.bool)
                evcurr_pos = curr_frame_pos.cpu().detach().numpy().astype(np.uint8)
                evcurr_neg = curr_frame_neg.cpu().detach().numpy().astype(np.uint8)
                saveevframe = evcurr_pos + evcurr_neg
                spk_evframe = torch.tensor(saveevframe != 0.00, dtype=torch.bool)

                spk_mask = torch.zeros_like(dens_mask).to(config.DEVICE)
                torch.logical_and(dens_mask.to(config.DEVICE), spk_evframe.to(config.DEVICE), out=spk_mask)
                num_evs_mask = torch.sum(spk_mask).item()
                num_evs_back = torch.sum(spk_evframe).item() - num_evs_mask
                try:
                    ratio = num_evs_back / num_evs_mask
                except ZeroDivisionError:
                    ratio = float('inf')  # or any other value that makes sense in your context

                print('ratio is: '+ str(ratio))
                if ratio < config.maxBackgroundRatio:
                    # Visualisation
                    OMS = OMS_pos + OMS_neg
                    OMS[OMS != 0] = 1.0 * 255

                    if config.SHOWIMGS:
                        cv2.imshow('Events map', spk_evframe.cpu().detach().numpy().astype(np.uint8) * 255)
                        cv2.imshow('mask', mask)
                        cv2.imshow('OMS map', OMS)
                        cv2.imshow('Saliency map OMS', cv2.applyColorMap(
                            (saliency_mapOMS).astype(np.uint8), cv2.COLORMAP_JET))
                        cv2.waitKey(1)

                    # Coordinates maximum value of saliency_mapOMS
                    max_coords = np.unravel_index(np.argmax(saliency_mapOMS), saliency_mapOMS.shape)
                    if mask[max_coords]!=0:
                        max_to_object+=1
                    plt.imsave( res_path+'evframes/' + f'evframe_{cnt}.png', spk_evframe.cpu().detach().numpy().astype(np.uint8) * 255, cmap='gray')
                    # plt.imsave(res_path + f'mask_{i}.png', spike_gt.cpu().numpy(), cmap='gray')
                    plt.imsave(res_path+'OMS/' + f'OMS_{cnt}.png', OMS, cmap='gray')
                    plt.imsave(res_path+'salmaps/' + f'salmap_{cnt}.png', saliency_mapOMS, cmap='jet')
                    print('frame: '+str(cnt))
                    cnt += 1
                i += 1
            time += time_wnd_frames
        accuracy.append(max_to_object / cnt * 100)
        with open(res_path + dir + '/' + seq_name + 'accuracy.pkl', 'wb') as f:
            pickle.dump(accuracy, f)
        print("accuracy "+ dir+'/'+seq_name+'/: '+ str())


