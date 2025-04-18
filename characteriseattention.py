import cv2
import numpy as np
from functions.data_helpers import load_yarp_events, create_results_folders
from functions.OMS_helpers import *
from functions.attention_helpers import AttentionModule
import matplotlib
matplotlib.use('TkAgg')




def process_events(events, attention_module, device, time_window):
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
        create_results_folders(config.RES_PATH, '/attentionOMS')

    cnt = 0
    for x, y, ts, pol in zip(e_x, e_y, e_ts, e_pol):
        if ts <= window_period:
            if pol == 1:
                window_pos[y][x] = 1.0
            else:
                window_neg[y][x] = 1.0
        else:
            #EVENTS
            window_combined = window_pos + window_neg
            window_combined[window_combined != 0] = 1.0 * 255
            #ATTENTION
            vSlice = np.expand_dims(np.stack((window_pos, window_neg)), 0)
            #OMS (split in polarities)
            wOMS_pos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(device)
            wOMS_neg = torch.tensor(window_neg, dtype=torch.float32).unsqueeze(0).to(device)
            OMSpos, indexes_pos = egomotion(wOMS_pos, net_center, net_surround, device, config.MAX_Y, config.MAX_X, config.OMS_PARAMS['threshold'])
            OMSneg, indexes_neg = egomotion(wOMS_neg, net_center, net_surround, device, config.MAX_Y, config.MAX_X, config.OMS_PARAMS['threshold'])
            OMSpos = OMSpos.squeeze(0).squeeze(0).cpu().detach().numpy()
            OMSneg = OMSneg.squeeze(0).squeeze(0).cpu().detach().numpy()
            vSliceOMS = np.expand_dims(np.stack((OMSpos, OMSneg)), 0)
            OMS = OMSpos + OMSneg
            OMS[OMS != 0] = 1.0 * 255
            with torch.no_grad():
                saliency_map = attention_module(
                    torch.tensor(vSlice, dtype=torch.float32).to(device)
                ).cpu().numpy()
            with torch.no_grad():
                saliency_mapOMS = attention_module(
                    torch.tensor(vSliceOMS, dtype=torch.float32).to(device)
                ).cpu().numpy()
            #visualisation
            saliency_map = saliency_map* 255
            saliency_mapOMS = saliency_mapOMS * 255

            cv2.imshow('Events map', window_combined)
            cv2.imshow('OMS map',OMS)
            cv2.imshow('Saliency map', cv2.applyColorMap(
                (saliency_map).astype(np.uint8), cv2.COLORMAP_JET))
            cv2.imshow('Saliency map OMS', cv2.applyColorMap(
                (saliency_mapOMS).astype(np.uint8), cv2.COLORMAP_JET))

            cv2.imwrite(f"{config.RES_PATH}events/Eventmap{cnt}.png", window_combined)
            cv2.imwrite(f"{config.RES_PATH}oms/OMSmap{cnt}.png", OMS)
            cv2.imwrite(f"{config.RES_PATH}attention/Saliencymap{cnt}.png", cv2.applyColorMap(
                (saliency_map).astype(np.uint8), cv2.COLORMAP_JET))
            cv2.imwrite(f"{config.RES_PATH}attentionOMS/SaliencymapOMS{cnt}.png", cv2.applyColorMap(
                (saliency_mapOMS).astype(np.uint8), cv2.COLORMAP_JET))

            cv2.waitKey(1)
            window_period = ts + time_window
            window_pos.fill(0)
            window_neg.fill(0)
            cnt += 1


class Config:
    # Constants
    MAX_X, MAX_Y = 304, 240
    RESOLUTION = (MAX_Y, MAX_X)
    CAMERA_EVENTS = 'right'
    CODEC = '20bit'

    # FILE_PATH = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calib_circles/calibration_circles/ATIS/'
    # FILE_PATH = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/v_frames/object_clutter2/ATIS/'
    # FILE_PATH = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/v_frames/object_clutter/ATIS/'
    FILE_PATH = '/Users/giuliadangelo/Downloads/attention-multiobjects/'

    # FILE_PATH = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calibration/no_obj/ATIS/'
    # FILE_PATH = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calibration/obj/ATIS/'
    EXP = FILE_PATH.split('/ATIS/')[0].split('/')[-1]
    RES_PATH = 'results/attention/'+EXP+'/'

    # OMS Parameters
    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.50,
        'tau_memOMS': 0.3,
        'sc':1,
        'ss':1
    }

    # Attention Parameters
    ATTENTION_PARAMS = {
        'VM_radius': 8, #(R0)
        'VM_radius_group': 15,
        'num_ori': 4,
        'b_inh': 3, #(w)
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2, #(rho)
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.3
    }
    DATA_PARAMS = {
        'TIME_WINDOW': 300
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
    process_events(events, net_attention, device, config.DATA_PARAMS['TIME_WINDOW'])
