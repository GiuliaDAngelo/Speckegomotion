import cv2
import numpy as np
from functions.OMS_helpers import *
import matplotlib
from bimvee.importIitYarp import importIitYarp
from functions.attention_simple_helpers import initialise_attention, run_attention


matplotlib.use('TkAgg')

def load_yarp_events(filePathOrName,codec,camera_events):
    # Load the event data using the 'importIitYarp' function
    # 'events' will contain the structured event data from the file
    events = importIitYarp(filePathOrName=filePathOrName,codec=codec)  # Codec to decode the event data
    # Extract the 'x' and 'y' coordinates of events, their timestamps ('ts'), and polarity ('pol')
    # These represent the x and y positions of the event, the time it occurred, and the polarity (whether it was a brightening or darkening event)
    e_x = events['data'][camera_events]['dvs']['x']  # x-coordinates of events
    e_y = events['data'][camera_events]['dvs']['y']  # y-coordinates of events
    e_ts = np.multiply(events['data'][camera_events]['dvs']['ts'], 10 ** 3)  # Convert timestamps to milliseconds
    e_pol = events['data'][camera_events]['dvs']['pol']  # Event polarity (1 for ON events, 0 for OFF events)
    return e_x,e_y,e_ts,e_pol


def process_events(events, attention_module, device, time_window):
    """Process events and display saliency maps."""
    e_x, e_y, e_ts, e_pol = events
    image_h, image_w = config.MAX_Y, config.MAX_X
    window_pos = np.zeros((image_h, image_w), dtype=np.float32)
    window_neg = np.zeros((image_h, image_w), dtype=np.float32)
    window_period = time_window


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
            OMS = OMSpos + OMSneg
            OMS[OMS != 0] = 1.0 * 255
            saliency_map, config.salmax_coords[:] = run_attention(OMS[0].detach().cpu().numpy(), net_attention, device, config.RESOLUTION,
                                                              config.size_krn_after_oms, num_pyr=3)
            #visualisation
            saliency_map = saliency_map* 255

            cv2.imshow('Events map', window_combined)
            cv2.imshow('OMS map',OMS.squeeze().squeeze().cpu().numpy())
            cv2.imshow('Saliency map', cv2.applyColorMap(
                (saliency_map).astype(np.uint8), cv2.COLORMAP_JET))

            cv2.waitKey(1)
            window_period = ts + time_window
            window_pos.fill(0)
            window_neg.fill(0)
            cnt += 1


class Config:
    # Constants
    MAX_X, MAX_Y = 304, 260
    RESOLUTION = (253, 297)
    CAMERA_EVENTS = 'right'
    CODEC = '24bit'

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
        'threshold': 0.70,
        'tau_memOMS': 0.3,
        'sc':1,
        'ss':1
    }

    # Attention Parameters
    ATTENTION_PARAMS = {
        'size_krn' : 10 , # Size of the kernel
        'r0' : 4 , # Radius shift from the center
        'rho' : 0.1 , # Scale coefficient to control arc length
        'theta' : np.pi * 3 / 2, # Angle to control the orientation of the arc
        'thetas' : np.arange(0, 2 * np.pi, np.pi / 4),
        'thick' : 3, # thickness of the arc
        'offsetpxs' : 0, # size / 2
        'offset' : (0, 0),
        'fltr_resize_perc' : [2, 2],
        'num_pyr' : 3,
        'tau_mem': 0.3,
        'stride':1,
        'out_ch':1
    }
    salmax_coords = np.zeros((2,), dtype=np.int32)
    size_krn_after_oms = 121
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
    net_attention = initialise_attention(device, config.ATTENTION_PARAMS)
    # Process and visualize events
    process_events(events, net_attention, device, config.DATA_PARAMS['TIME_WINDOW'])