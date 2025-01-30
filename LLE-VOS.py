import numpy as np
import cv2
import torch
from functions.attention_helpers import AttentionModule
import os
from functions.OMS_helpers import initialize_oms, egomotion
import matplotlib.pyplot as plt
import natsort


# from 60


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
        'VM_radius': 4,  # (R0)
        'VM_radius_group': 10,
        'num_ori': 4,
        'b_inh': 3,  # (w)
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2,  # (rho)
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.1
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

if __name__ == '__main__':

    directory = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO2LowightChallengingConditions/'
    events_dir = directory+ 'events/'
    annotations_dir = directory + 'annotations/'
    ev_frames_dir = directory + 'ev_frames/'
    oms_dir = directory + 'oms/'
    attention_dir = directory + 'attention/'


    dirs_events = [d for d in os.listdir(events_dir) if os.path.isdir(os.path.join(events_dir, d))]
    dirs_events = sorted(dirs_events)


    config = Config()
    # Initialize OMS network
    net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)

    # Initialize Attention modules
    net_attention = AttentionModule(**config.ATTENTION_PARAMS)
    mean_accuracy = []
    for dir in dirs_events:
        ann_files = [f for f in os.listdir(annotations_dir + dir) if f.endswith('.png')]
        ann_files = sorted(ann_files)
        accuracy = []
        bbox = 10
        max_to_object = []
        cnt = 1

        mkdirfold(oms_dir + dir)
        mkdirfold(attention_dir + dir)

        for ann_file in ann_files:
            ann_img = cv2.imread(annotations_dir + dir + '/' + ann_file, cv2.IMREAD_GRAYSCALE)
            ev_frames_pos = [f for f in os.listdir(ev_frames_dir + dir + '/pos/') if f.endswith('.png') and ann_file.split('.png')[0] in f]
            ev_frames_pos = natsort.natsorted(ev_frames_pos)
            ev_frames_neg = [f for f in os.listdir(ev_frames_dir + dir + '/neg/') if
                             f.endswith('.png') and ann_file.split('.png')[0] in f]
            ev_frames_neg = natsort.natsorted(ev_frames_neg)
            for ev_frame_pos, ev_frame_neg in zip(ev_frames_pos, ev_frames_neg):
                window_pos = torch.tensor(
                    cv2.imread(ev_frames_dir + dir + '/pos/' + ev_frame_pos, cv2.IMREAD_GRAYSCALE), dtype=torch.uint8)
                window_neg = torch.tensor(
                    cv2.imread(ev_frames_dir + dir + '/neg/' + ev_frame_neg, cv2.IMREAD_GRAYSCALE), dtype=torch.uint8)
                # compute egomotion
                OMS_pos, indexes_pos = egomotion(window_pos, net_center, net_surround, config.DEVICE, window_pos.shape[0], window_pos.shape[1],
                                                 config.OMS_PARAMS['threshold'])
                OMS_neg, indexes_neg = egomotion(window_neg, net_center, net_surround, config.DEVICE, window_neg.shape[0], window_neg.shape[1],
                                                 config.OMS_PARAMS['threshold'])
                # compute attention
                OMS_pos = OMS_pos.squeeze(0).squeeze(0).cpu().detach().numpy()
                OMS_neg = OMS_neg.squeeze(0).squeeze(0).cpu().detach().numpy()
                vSliceOMS = np.expand_dims(np.stack((OMS_pos, OMS_neg)), 0)
                with torch.no_grad():
                    saliency_mapOMS = (net_attention(
                        torch.tensor(vSliceOMS, dtype=torch.float32).to(config.DEVICE)
                    ).cpu().numpy()) * 255

                # show event map and annotation map
                # window = window_pos + window_neg
                # window[window != 0] = 1.0 * 255
                # cv2.imshow('Event neg map', window.cpu().numpy())
                # cv2.imshow('Annotation map', ann_img)
                # cv2.imshow('Saliency map OMS', cv2.applyColorMap(
                #     (saliency_mapOMS).astype(np.uint8), cv2.COLORMAP_JET))
                # Visualisation
                OMS = OMS_pos + OMS_neg
                OMS[OMS != 0] = 1.0 * 255
                # cv2.imshow('OMS', OMS)
                # cv2.waitKey(1)


                # Resize the saliency_mapOMS to match the dimensions of the mask
                saliency_mapOMS = cv2.resize(saliency_mapOMS, (ann_img.shape[1], ann_img.shape[0]))
                # need to extract the coordinates of the max value in the saliency map
                max_coords = np.unravel_index(np.argmax(saliency_mapOMS, axis=None), saliency_mapOMS.shape)

                max_saliency_mapOMS = torch.zeros_like(torch.tensor(saliency_mapOMS)).to(config.DEVICE)
                max_saliency_mapOMS[max_coords[0] - (bbox // 2):max_coords[0] + (bbox // 2),
                max_coords[1] - (bbox // 2):max_coords[1] + (bbox // 2)] = 1
                spk_acc = torch.zeros_like(torch.tensor(ann_img), dtype=torch.bool).to(config.DEVICE)
                torch.logical_and(torch.tensor(ann_img, dtype=torch.bool).to(config.DEVICE),
                                  max_saliency_mapOMS, out=spk_acc)
                print(torch.sum(spk_acc).item())

                if torch.sum(spk_acc).item() != 0:
                    max_to_object.append(1)
                else:
                    max_to_object.append(0)

                plt.imsave(oms_dir + dir + f'/OMS_{cnt}.png', OMS, cmap='gray')
                plt.imsave(attention_dir + dir + f'/salmap_{cnt}.png', saliency_mapOMS, cmap='jet')

                print('frame: '+str(cnt))
                cnt+=1
        if len(max_to_object)!=0:
            dir_seq_acc = max_to_object.count(1) / len(max_to_object) * 100
        with open(os.path.join(attention_dir + dir, f'{dir}_accuracy.txt'), 'w') as f:
            f.write(f"accuracy {dir}: {dir_seq_acc}\n")
        print("accuracy " + dir + ': ' + str(dir_seq_acc))
        mean_accuracy.append(dir_seq_acc)
    print('total accuracy mean : ' + str(np.mean(mean_accuracy)))
    print('total accuracy std : ' + str(np.std(mean_accuracy)))