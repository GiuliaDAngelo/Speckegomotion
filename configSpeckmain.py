import time
import numpy as np


# Visualization parameters
resolution = [128, 128] # Resolution of the DVS sensor
max_x = resolution[0]
max_y = resolution[1]
drop_rate = 0.0  # Percentage of events to drop
<<<<<<< HEAD
update_interval = 0.001 #0.02 #seconds
=======
update_interval = 0.02  # Update every 0.02 seconds
>>>>>>> ad813c4 (WIP attention in a separate thread)
last_update_time = time.time()


# Parameters kernel
size_krn_center = 8  # Size of the kernel (NxN)
sigma_center = 1  # Sigma for the first Gaussian
size_krn_surround = 8  # Size of the kernel (NxN)
sigma_surround = 4  # Sigma for the first Gaussian

num_pyr = 1
tau_mem = 0.01
threshold = 200


# Visual attention paramters
size = 10  # Size of the kernel
r0 = 4  # Radius shift from the center
rho = 0.1  # Scale coefficient to control arc length
theta = np.pi * 3 / 2  # Angle to control the orientation of the arc
thick = 3  # thickness of the arc
offsetpxs = 0 #size / 2
offset = (offsetpxs, offsetpxs)
fltr_resize_perc = [2, 2]
num_pyr = 3

# Create Von Mises (VM) filters with specified parameters
# The angles are generated in radians, ranging from 0 to 2π in steps of π/4
thetas = np.arange(0, 2 * np.pi, np.pi / 4)