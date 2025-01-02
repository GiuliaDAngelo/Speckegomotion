import cv2
import torch
import numpy as np
from functions.data_helpers import load_yarp_events, create_results_folders
from functions.OMS_helpers import *
from functions.attention_helpers import AttentionModule
import matplotlib
matplotlib.use('TkAgg')




def process_events(events, attention_module, device, time_window=300):
    """Process events and display saliency maps."""
    e_x, e_y, e_ts, e_pol = events
    image_h, image_w = config.MAX_Y, config.MAX_X
    window_pos = np.zeros((image_h, image_w), dtype=np.float32)
    window_neg = np.zeros((image_h, image_w), dtype=np.float32)
    window_period = time_window

    # Create results folders
    if config.RES_PATH:
        create_results_folders(config.RES_PATH, '/events')
        create_results_folders(config.RES_PATH, '/oms')
        create_results_folders(config.RES_PATH, '/attention')

    cnt = 0
    for x, y, ts, pol in zip(e_x, e_y, e_ts, e_pol):
        if ts <= window_period:
            if pol == 1:
                window_pos[y][x] = 1.0
            else:
                window_neg[y][x] = 1.0
        else:
            #OMS
            window_combined = window_pos + window_neg
            window_combined[window_combined != 0] = 1.0
            window_OMS = torch.tensor(window_combined, dtype=torch.float32).unsqueeze(0).to(device)
            OMS, indexes = egomotion(window_OMS, net_center, net_surround, device, config.MAX_Y, config.MAX_X, config.OMS_PARAMS['threshold'])
            #ATTENTION
            vSlice = np.expand_dims(np.stack((window_pos, window_neg)), 0)
            with torch.no_grad():
                saliency_map = attention_module(
                    torch.tensor(vSlice, dtype=torch.float32).to(device)
                ).cpu().numpy()
            #visualisation
            window_combined = window_combined * 255
            cv2.imshow('Events map', window_combined)
            cv2.imshow('OMS map',OMS[0][0].cpu().detach().numpy())
            cv2.imshow('Saliency map', cv2.applyColorMap(
                (saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET))
            cv2.imwrite(f"{config.RES_PATH}events/Eventmap{cnt}.png", window_combined)
            cv2.imwrite(f"{config.RES_PATH}oms/OMSmap{cnt}.png", OMS[0].cpu().numpy().squeeze())
            cv2.imwrite(f"{config.RES_PATH}attention/Saliencymap{cnt}.png", cv2.applyColorMap(
                (saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET))
            cv2.waitKey(1)
            window_period += time_window
            window_pos.fill(0)
            window_neg.fill(0)
            cnt += 1


class Config:
    # Constants
    MAX_X, MAX_Y = 304, 240
    RESOLUTION = (MAX_Y, MAX_X)
    CAMERA_EVENTS = 'right'
    CODEC = '20bit'

    EXP = 'no_obj'
    RES_PATH = 'results/attention/'+EXP+'/'
    # FILE_PATH = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calib_circles/calibration_circles/ATIS/'
    FILE_PATH = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calibration/no_obj/ATIS/'
    # FILE_PATH = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calibration/obj/ATIS/'


    # OMS Parameters
    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.50,
        'tau_memOMS': 0.1,
        'sc':1,
        'ss':1
    }

    # Attention Parameters
    ATTENTION_PARAMS = {
        'VM_radius': 8, #(R0)
        'VM_radius_group': 15,
        'num_ori': 4,
        'b_inh': 1.5,
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2, #(rho)
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.1
    }





if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    config = Config()
    # Load data
    events = load_yarp_events(config.FILE_PATH, config.CODEC, config.CAMERA_EVENTS)
    # Initialize OMS
    net_center, net_surround = initialize_oms(device, config.OMS_PARAMS)
    # Initialize Attention modules
    net_attention = AttentionModule(**config.ATTENTION_PARAMS)
    # Process and visualize events
    process_events(events, net_attention, device)
