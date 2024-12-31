# Import the function 'importIitYarp' from the 'bimvee' library to load event-based camera data
import matplotlib
import torch
import cv2
import os
import yaml
from functions.data_helpers import load_yarp_events
from functions.OMS_helpers import *
from functions.attention_helpers import *
matplotlib.use('TkAgg')



##########################
####### Parameters #######
##########################

# Define the dimensions of the event-based camera (304 pixels width by 240 pixels height)
max_x = 304
max_y = 240
resolution = (max_y, max_x)

# Parameters OMS
size_krn_center = 8  # Size of the kernel (NxN) (all half ) - 8
sigma_center = 1  # Sigma for the first Gaussian - 1
size_krn_surround = 8  # Size of the kernel (NxN) - 8
sigma_surround = 4  # Sigma for the first Gaussian - 4
threshold = 0.50
tau_memOMS = 0.1  # Time constant for the membrane potential

#Parameters attention
VM_radius=10
VM_radius_group=15
num_ori=4
b_inh=1.5
g_inh=1.0
w_sum=0.5
vm_w=.1
vm_w2=.4
vm_w_group=.2
vm_w2_group=.4
random_init=False
lif_tau=0.1



if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    #####################
    ##### load data #####
    #####################
    camera_events = 'right'
    codec = '20bit'
    # codec = '24bit'

    # filePathOrName = '/Users/giuliadangelo/workspace/data/DATASETs/attention-multiobjects/'
    # filePathOrName = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calibration/obj/ATIS/'
    filePathOrName = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calib_circles/calibration_circles/ATIS/'
    [e_x,e_y,e_ts,e_pol] = load_yarp_events(filePathOrName,codec,camera_events)

    #####################
    ##### egomotion #####
    #####################
    center, surround = OMSkernels(size_krn_center, sigma_center, size_krn_surround, sigma_surround)
    ss = 1
    sc = 1 #ss + sigma_surround - sigma_center
    net_center = net_def(center, tau_memOMS, 1, 1, size_krn_center, device, sc)
    net_surround = net_def(surround, tau_memOMS, 1, 1, size_krn_surround, device, ss)

    #####################
    ##### attention #####
    #####################
    # Initialize the attention module with parameters from the YAML file
    am = AttentionModule(VM_radius=VM_radius,
                         VM_radius_group=VM_radius_group,
                         num_ori=num_ori,
                         b_inh=b_inh,
                         g_inh=g_inh,
                         w_sum=w_sum,
                         vm_w=vm_w,
                         vm_w2=vm_w2,
                         vm_w_group=vm_w_group,
                         vm_w2_group=vm_w2_group,
                         random_init=random_init,
                         lif_tau=lif_tau)

    # Set up parameters for processing events
    time_window = 400
    window_period = time_window
    image_w = 304
    image_h = 240
    window_pos = np.zeros((image_h, image_w), dtype=np.float32)
    window_neg = np.zeros((image_h, image_w), dtype=np.float32)

    # Process events and generate saliency maps
    for x, y, ts, pol in zip(e_x, e_y, e_ts, e_pol):
        if ts <= window_period:
            if pol == 1:
                window_pos[y][x] = 1.0
            else:
                window_neg[y][x] = 1.0
        else:
            vSlice = np.expand_dims(np.stack((window_pos, window_neg)), 0)
            with torch.no_grad():
                saliency_map = am(torch.tensor(vSlice,
                                               dtype=torch.float32,
                                               requires_grad=False).to(device)).cpu().numpy()

            cv2.imshow('salmap', cv2.applyColorMap((saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET))
            cv2.waitKey(1)
            window_period += time_window
            window_pos = np.zeros((image_h, image_w), dtype=np.float32)
            window_neg = np.zeros((image_h, image_w), dtype=np.float32)