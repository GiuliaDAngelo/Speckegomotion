import time


# Visualization parameters
resolution = [128, 128] # Resolution of the DVS sensor
max_x = resolution[0]
max_y = resolution[1]
drop_rate = 0.6  # Percentage of events to drop
update_interval = 0.02  # Update every 0.02 seconds
last_update_time = time.time()


# Parameters kernel
size_krn_center = 8  # Size of the kernel (NxN)
sigma_center = 1  # Sigma for the first Gaussian
size_krn_surround = 8  # Size of the kernel (NxN)
sigma_surround = 4  # Sigma for the first Gaussian

num_pyr = 1
tau_mem = 0.01
threshold = 200