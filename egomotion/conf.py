# --------------------------------------
import sys

# --------------------------------------
import time

# --------------------------------------
import numpy as np

# --------------------------------------
import matplotlib

matplotlib.use("TkAgg")

# --------------------------------------
import torch

# --------------------------------------
from datetime import datetime

# --------------------------------------
from pathlib import Path

# --------------------------------------
from environs import Env

# --------------------------------------
from loguru import logger

# --------------------------------------
from egomotion.utils import mkdir

# Environment variables
# ==================================================
env = Env(expand_vars=True)
env.read_env()

# Core paths
# ==================================================
ROOT_DIR = Path(__file__).parent.parent
IN_DIR = mkdir(env.path("IN_DIR", ROOT_DIR / "data/input"))
OUT_DIR = mkdir(env.path("OUT_DIR", ROOT_DIR / "data/output"))
PLOT_DIR = mkdir(env.path("PLOT_DIR", ROOT_DIR / "data/output/plots"))

# Logger configuration
# ==================================================
# Enable colour tags in messages.
logger = logger.opt(colors=True)

LOG_LEVEL = env.str("LOG_LEVEL", "INFO")

#: Log format.
log_config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": "<magenta>Egomotion</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>",
            "level": LOG_LEVEL,
        }
    ]
}

logger.configure(**log_config)

# filePathOrName = respath_to_file.get(respath, None)

# Kernel parameters
# REVIEW: This is superceded by the parameters below
# k_size = 8  # Size of the kernel (NxN)
# k_sigma = 5  # Sigma for the first Gaussian

k_center_size = 8  # Size of the kernel (NxN)
k_center_sigma = 1  # Sigma for the first Gaussian
k_surround_size = 8  # Size of the kernel (NxN)
k_surround_sigma = 4  # Sigma for the second Gaussian

# Event parameters
# ==================================================
ev_polarity = True
ev_video_fps = 60.0  # Frames per second
ev_video_dur = 4  # Duration of video in seconds
ev_ts_ms = False  # Flag to convert timestamps to microseconds

# Stimulus parameters
# ==================================================
stim_speed = 60.0  # Speed of stimulus in pixels per second/ 1000 ms
stim_show_egomap = True
stim_save_res = True
# REVIEW: This is defined multiple times
vm_num_pyr = 1
threshold = 0

# Visual attention parameters (von Mises filters)
# REVIEW: This is defined multiple times
# ==================================================
vm_num_pyr = 1  # Number of pyramids
tau_mem = 0.01  # Membrane time constant
threshold = 200

vm_size = 10  # Size of the kernel
vm_r0 = 4  # Radius shift from the center
vm_rho = 0.1  # Scale coefficient to control arc length
vm_theta = np.pi * 3 / 2  # Angle to control the orientation of the arc
vm_thick = 3  # thickness of the arc
vm_offsetpxs = 0  # size / 2
vm_offset = (vm_offsetpxs, vm_offsetpxs)
vm_fltr_resize_perc = [2, 2]
vm_num_pyr = 3

# The angles are generated in radians,
# ranging from 0 to 2π in steps of π/4
vm_thetas = np.arange(0, 2 * np.pi, np.pi / 4)


# Visualization parameters
# ==================================================
vis_resolution = [128, 128]  # Resolution of the DVS sensor
vis_max_x = vis_resolution[0]
vis_max_y = vis_resolution[1]
vis_drop_rate = 0.0  # Percentage of events to drop
vis_update_interval = 0.001  # 0.02 #seconds
last_update_time = datetime.now()

# PyTorch parameters
# ==================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Dummy operation - for debugging purposes
# ==================================================
dummy = False
