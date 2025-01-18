
import cv2
import numpy as np
from functions.OMS_helpers import *
from functions.attention_helpers import AttentionModule
from functions.Speck_helpers import Specksetup
import matplotlib
import time

matplotlib.use('TkAgg')


class Config:
    # Constants
    RESOLUTION = [128, 128]  # Resolution of the DVS sensor
    MAX_X = RESOLUTION[0]
    MAX_Y = RESOLUTION[1]
    DROP_RATE = 0.3  # Percentage of events to drop
    UPDATE_INTERVAL = 0.02  # 0.02 #seconds
    LAST_UPDATE_INTERVAL = time.time()
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


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


def compute_OMS(window_pos,window_neg):
    # compute OMS maps
    OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    OMSneg = torch.tensor(window_neg, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

    OMSpos_map, indexes_pos = egomotion(OMSpos, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
                                        config.OMS_PARAMS['threshold'])
    OMSneg_map, indexes_neg = egomotion(OMSneg, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
                                        config.OMS_PARAMS['threshold'])

    OMSpos_map = OMSpos_map.squeeze(0).squeeze(0).cpu().detach().numpy()
    OMSneg_map = OMSneg_map.squeeze(0).squeeze(0).cpu().detach().numpy()

    cv2.imshow('Events pos map', window_pos)
    cv2.imshow('Events neg map', window_neg)
    cv2.imshow('OMS pos map', OMSpos_map)
    cv2.imshow('OMS neg map', OMSneg_map)
    cv2.waitKey(1)
    return OMSpos_map, OMSneg_map

def compute_attention(OMSpos_map,OMSneg_map):
    vSlice = np.expand_dims(np.stack((OMSpos_map, OMSneg_map)), 0)
    with torch.no_grad():
        saliency_map = net_attention(
            torch.tensor(vSlice, dtype=torch.float32).to(config.DEVICE)
        ).cpu().numpy()
    cv2.imshow('Saliency map', saliency_map*255)
    return saliency_map

if __name__ == '__main__':
    config = Config()

    # Initialize OMS
    net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)
    # Initialize Attention modules
    net_attention = AttentionModule(**config.ATTENTION_PARAMS)
    # Initialise Speck
    sink, window_pos, window_neg, numevs, events_lock = Specksetup(config.RESOLUTION, config.DROP_RATE)

    last_update_time = time.time()
    while True:
        current_time = time.time()
        with events_lock:
            if current_time - last_update_time > config.UPDATE_INTERVAL:
                if numevs[0] > 0:
                    # compute OMS
                    OMSpos_map, OMSneg_map = compute_OMS(window_pos, window_neg)
                    # compute attention
                    saliency_map = compute_attention(OMSpos_map, OMSneg_map)
                    window_pos.fill(0)
                    window_neg.fill(0)
                last_update_time = current_time