import time


# Visualization parameters
resolution = [128, 128] # Resolution of the DVS sensor
max_x = resolution[0]
max_y = resolution[1]
drop_rate = 0.6  # Percentage of events to drop
update_interval = 0.02  # Update every 0.02 seconds
last_update_time = time.time()


# Parameters kernel
size_krn = 8  # Size of the kernel (NxN)
sigma = 4  # Sigma for the first Gaussian
num_pyr = 1
tau_mem = 0.03
threshold = 180