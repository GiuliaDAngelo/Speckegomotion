# Model developed by: Giulia D'Angelo
# Summary: Processes event-based data using a spiking neural network to generate and save object-motion maps.

from functions.OMS_helpers import initialize_oms, egomotion
from functions.data_helpers import load_eventsnpy, create_video_from_frames,create_results_folders
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

# Configuration
class Config:
    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.80,
        'tau_memOMS': 0.1,
        'sc': 1,
        'ss': 1
    }
    POLARITY = True
    TS_FLAG = False
    SHOW_EGOMAP = True
    SAVE_RES = True
    CHARACTERISATION_FLAG = True
    EXP = 'objego'
    SF = 0.3
    SF_SMALL = 3
    SPEED = 0.01
    SMALL_SPEED = 0.09
    DURATION = 2
    FPS_VIDEO = 60.0
    BASE_PATH = '/Users/giuliadangelo/workspace/code/IEBCS/data/video/stimuli/'
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# Utility Functions
def setup_paths(config: Config):
    name_exp = f"{config.EXP}sf{config.SF}sp{config.SPEED}sf{config.SF_SMALL}sp{config.SMALL_SPEED}".replace(".", "")
    if config.CHARACTERISATION_FLAG:
        exp_kernel = f"krncenter{config.OMS_PARAMS['size_krn_center']}sigcenter{config.OMS_PARAMS['sigma_center']}krnsurround{config.OMS_PARAMS['size_krn_surround']}sigsurround{config.OMS_PARAMS['sigma_surround']}"
    else:
        exp_kernel = ""
    respath = f"results/{name_exp}{exp_kernel}/"
    evdata_path = "ev_100_10_100_300_0.3_0.01.dat"
    evpath = f"{config.BASE_PATH}{name_exp}/{name_exp}{evdata_path}.npy"
    return respath, evpath


# Main Processing Function
def process_events(config: Config):
    respath, evpath = setup_paths(config)

    # Create results folders
    if config.SAVE_RES:
        create_results_folders(respath)

    # Load events
    time_wnd_frames = 1000  # 1kHz as IEBCS
    frames, max_y, max_x = load_eventsnpy(config.POLARITY, config.DURATION, config.FPS_VIDEO, evpath, config.TS_FLAG, time_wnd_frames)
    len_fr = len(frames)
    time_wnd_frames = config.DURATION / len_fr

    # Initialize egomotion network
    net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)

    # Process frames
    spikes = [[] for _ in range((max_x + 1) * (max_y + 1))]
    time = 0

    for cnt, frame in enumerate(frames):
        time += time_wnd_frames
        print(f"{cnt} frame out of {len(frames)}")
        OMS, indexes = egomotion(frame[0], net_center, net_surround, config.DEVICE, max_y, max_x, config.OMS_PARAMS['threshold'])

        # Extract spikes
        indtrue = np.argwhere(indexes[0].cpu() == True)
        for i in range(len(indtrue[0])):
            x, y = indtrue[1][i], indtrue[0][i]
            spikes[x + (y * max_x)].append(time)

        # Show the egomap
        if config.SHOW_EGOMAP:
            plt.clf()
            plt.imshow(OMS[0].cpu().numpy(), cmap='gray', vmin=0, vmax=255)
            plt.colorbar(shrink=0.3)
            plt.draw()
            plt.pause(0.001)

        # Save results
        if config.SAVE_RES:
            cv2.imwrite(f"{respath}OMS/OMSmap{cnt}.png", OMS[0].cpu().numpy())

    # Save spikes
    with open(f"{respath}spikes.pkl", 'wb') as f:
        pickle.dump(spikes, f)

    # Save videos
    if config.SAVE_RES:
        create_video_from_frames(f"{respath}OMS/", f"{respath}{respath.split('/')[1]}OMSmap.mp4", fps=config.FPS_VIDEO)

    print("Processing complete.")


# Entry Point
if __name__ == '__main__':
    config = Config()
    process_events(config)
