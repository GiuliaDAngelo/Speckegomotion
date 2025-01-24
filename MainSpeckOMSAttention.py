import cv2
import numpy as np
import torch

from functions.OMS_helpers import *
from functions.attention_helpers import AttentionModule
from functions.Speck_helpers import Specksetup
import matplotlib
import time
import matplotlib.pyplot as plt

# Use 'Agg' backend for matplotlib to avoid Tkinter issues
matplotlib.use('Agg')

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

def compute_OMS(window_pos):
    window_pos = np.flipud(window_pos).copy()
    # compute OMS maps
    OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

    OMSpos_map, indexes_pos = egomotion(OMSpos, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
                                        config.OMS_PARAMS['threshold'])

    OMSpos_map = OMSpos_map.squeeze(0).squeeze(0).cpu().detach().numpy()
    return OMSpos_map, indexes_pos

def compute_OMSpol(window_pos, window_neg):
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

def compute_attention(OMSpos_map, OMSneg_map):
    vSlice = np.expand_dims(np.stack((OMSpos_map, OMSneg_map)), 0)
    with torch.no_grad():
        saliency_map = net_attention(
            torch.tensor(vSlice, dtype=torch.float32).to(config.DEVICE)
        ).cpu().numpy()
    cv2.imshow('Saliency map', saliency_map * 255)
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
    times = []
    ev_window_counts = []
    ev_oms_counts = []
    plt.ion()
    start_time = time.time()
    sim_time = 40
    showstats = 1
    while True:
        current_time = time.time()
        with events_lock:
            if current_time - last_update_time > config.UPDATE_INTERVAL:
                if numevs[0] > 0:
                    OMS, indexes = compute_OMS(window_pos)
                    cv2.imshow('Events pos map', window_pos)
                    cv2.imshow('OMS', OMS)
                    cv2.waitKey(1)

                    if showstats:
                        plt.plot([current_time], [numevs[0]], 'ro-', label='Events')
                        plt.plot([current_time], [indexes.sum().item()], 'bo-', label='Events after suppression')
                        plt.plot([current_time], [numevs[0] - indexes.sum().item()], 'yo-', label='Events dropping')
                        plt.title('Comparison of Events before and after suppression')
                        plt.xlabel('Time')
                        plt.ylabel('Events Count')
                        if not plt.gca().get_legend():
                            plt.legend()
                        plt.pause(0.001)  # Pause to update the figure

                    window_pos.fill(0)
                    window_neg.fill(0)
                last_update_time = current_time



#
# import cv2
# import numpy as np
# import torch
#
# from functions.OMS_helpers import *
# from functions.attention_helpers import AttentionModule
# from functions.Speck_helpers import Specksetup
# import matplotlib
# import time
# import matplotlib.pyplot as plt
#
# matplotlib.use('Agg')
#
#
# class Config:
#     # Constants
#     RESOLUTION = [128, 128]  # Resolution of the DVS sensor
#     MAX_X = RESOLUTION[0]
#     MAX_Y = RESOLUTION[1]
#     DROP_RATE = 0.3  # Percentage of events to drop
#     UPDATE_INTERVAL = 0.02  # 0.02 #seconds
#     LAST_UPDATE_INTERVAL = time.time()
#     DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#
#
#     # OMS Parameters
#     OMS_PARAMS = {
#         'size_krn_center': 8,
#         'sigma_center': 1,
#         'size_krn_surround': 8,
#         'sigma_surround': 4,
#         'threshold': 0.50,
#         'tau_memOMS': 0.3,
#         'sc':1,
#         'ss':1
#     }
#
#     # Attention Parameters
#     ATTENTION_PARAMS = {
#         'VM_radius': 8, #(R0)
#         'VM_radius_group': 15,
#         'num_ori': 4,
#         'b_inh': 3, #(w)
#         'g_inh': 1.0,
#         'w_sum': 0.5,
#         'vm_w': 0.2, #(rho)
#         'vm_w2': 0.4,
#         'vm_w_group': 0.2,
#         'vm_w2_group': 0.4,
#         'random_init': False,
#         'lif_tau': 0.3
#     }
#
# def compute_OMS(window_pos):
#     window_pos = np.flipud(window_pos).copy()
#     # compute OMS maps
#     OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
#
#     OMSpos_map, indexes_pos = egomotion(OMSpos, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
#                                         config.OMS_PARAMS['threshold'])
#
#     OMSpos_map = OMSpos_map.squeeze(0).squeeze(0).cpu().detach().numpy()
#     return OMSpos_map, indexes_pos
#
# def compute_OMSpol(window_pos,window_neg):
#     # compute OMS maps
#     OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
#     OMSneg = torch.tensor(window_neg, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
#
#     OMSpos_map, indexes_pos = egomotion(OMSpos, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
#                                         config.OMS_PARAMS['threshold'])
#     OMSneg_map, indexes_neg = egomotion(OMSneg, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
#                                         config.OMS_PARAMS['threshold'])
#
#     OMSpos_map = OMSpos_map.squeeze(0).squeeze(0).cpu().detach().numpy()
#     OMSneg_map = OMSneg_map.squeeze(0).squeeze(0).cpu().detach().numpy()
#
#     cv2.imshow('Events pos map', window_pos)
#     cv2.imshow('Events neg map', window_neg)
#     cv2.imshow('OMS pos map', OMSpos_map)
#     cv2.imshow('OMS neg map', OMSneg_map)
#     cv2.waitKey(1)
#     return OMSpos_map, OMSneg_map
#
# def compute_attention(OMSpos_map,OMSneg_map):
#     vSlice = np.expand_dims(np.stack((OMSpos_map, OMSneg_map)), 0)
#     with torch.no_grad():
#         saliency_map = net_attention(
#             torch.tensor(vSlice, dtype=torch.float32).to(config.DEVICE)
#         ).cpu().numpy()
#     cv2.imshow('Saliency map', saliency_map*255)
#     return saliency_map
#
# if __name__ == '__main__':
#     config = Config()
#
#     # Initialize OMS
#     net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)
#     # Initialize Attention modules
#     net_attention = AttentionModule(**config.ATTENTION_PARAMS)
#     # Initialise Speck
#     sink, window_pos, window_neg, numevs, events_lock = Specksetup(config.RESOLUTION, config.DROP_RATE)
#
#     last_update_time = time.time()
#     times = []
#     ev_window_counts = []
#     ev_oms_counts = []
#     plt.ion()
#     # fig, ax = plt.subplots()
#     # ax.set_xlabel('Time (s)')
#     # ax.set_ylabel('Event Count')
#     # ax.legend()
#     start_time = time.time()
#     sim_time = 40
#     showstats = 1
#     # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
#     while True:
#         current_time = time.time()
#         with events_lock:
#             if current_time - last_update_time > config.UPDATE_INTERVAL:
#                 if numevs[0] > 0:
#                     OMS, indexes = compute_OMS(window_pos)
#                     cv2.imshow('Events pos map', window_pos)
#                     cv2.imshow('OMS', OMS)
#                     cv2.waitKey(1)
#
#                     if showstats:
#                         # print number of events
#                         # print('Number of events: ' + str(numevs[0]))
#                         # print('Number of suprressed events:', indexes.sum().item())
#                         plt.plot([current_time], [numevs[0]], 'ro-', label='Events')
#                         plt.plot([current_time], [indexes.sum().item()], 'bo-', label='Events after suppression')
#                         plt.plot([current_time], [numevs[0] - indexes.sum().item()], 'yo-', label='Events dropping')
#                         plt.title('Comparison of Events before and after suppression')
#                         plt.xlabel('Time')
#                         plt.ylabel('Events Count')
#                         if not plt.gca().get_legend():
#                             plt.legend()
#                         plt.pause(0.001)  # Pause to update the figure
#
#                     # ev_window = np.count_nonzero(window_pos)
#                     # ev_oms = np.count_nonzero(OMS)
#                     # print(f'mean Events count: {ev_window}')
#                     # print(f'mean OMS events: {ev_oms}')
#                     #
#                     #
#                     # # Append the current time and event counts to the lists
#                     # if current_time - start_time > sim_time:
#                     #     print(f'mean Events count: {np.mean(ev_window_counts):.2f}')
#                     #     print(f'std Events count: {np.std(ev_window_counts):.2f}')
#                     #     print(f'mean OMS events: {np.mean(ev_oms_counts):.2f}')
#                     #     print(f'std OMS events: {np.std(ev_oms_counts):.2f}')
#                     #     break
#                     # times.append(current_time - start_time)
#                     # ev_window_counts.append(ev_window)
#                     # ev_oms_counts.append(ev_oms)
#
#                     # ax1.imshow(window_pos, cmap='gray')
#                     # ax1.set_title('Events map')
#                     # ax2.imshow(OMS, cmap='gray')
#                     # ax2.set_title('OMS map')
#                     # ax3.scatter(times, ev_window_counts, label='Events', color='blue')
#                     # ax3.scatter(times, ev_oms_counts, label='OMS events', color='orange')
#                     # ax3.legend(['Events', 'OMS events'])
#                     # ax3.relim()
#                     # ax3.autoscale_view()
#                     # plt.draw()
#                     # plt.pause(0.01)
#
#                     # compute OMS
#                     # OMSpos_map, OMSneg_map = compute_OMSpol(window_pos, window_neg)
#                     # # compute attention
#                     # saliency_map = compute_attention(OMSpos_map, OMSneg_map)
#                     window_pos.fill(0)
#                     window_neg.fill(0)
#                 last_update_time = current_time