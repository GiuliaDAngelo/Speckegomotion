# Import the function 'importIitYarp' from the 'bimvee' library to load event-based camera data
from bimvee.importIitYarp import importIitYarp
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from functions.OMS_helpers import *


#############################
######### load data #########
#############################


# Define the camera from which events will be extracted ('right' camera)
camera_events = 'right'

# Specify the codec format used for decoding the event data ('24bit' format)
codec = '20bit'
# codec = '24bit'


# Set the file path where the event data is stored
# filePathOrName = '/Users/giuliadangelo/workspace/data/DATASETs/attention-multiobjects/'
# filePathOrName = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calibration/obj/ATIS/'
filePathOrName = '/Users/giuliadangelo/workspace/data/DATASETs/IROS_attention/calib_circles/calibration_circles/ATIS/'

# Load the event data using the 'importIitYarp' function
# 'events' will contain the structured event data from the file
events = importIitYarp(
    filePathOrName=filePathOrName,  # Path to the dataset
    codec=codec)  # Codec to decode the event data

# Extract the 'x' and 'y' coordinates of events, their timestamps ('ts'), and polarity ('pol')
# These represent the x and y positions of the event, the time it occurred, and the polarity (whether it was a brightening or darkening event)
e_x = events['data'][camera_events]['dvs']['x']  # x-coordinates of events
e_y = events['data'][camera_events]['dvs']['y']  # y-coordinates of events
e_ts = np.multiply(events['data'][camera_events]['dvs']['ts'], 10 ** 3)  # Convert timestamps to milliseconds
e_pol = events['data'][camera_events]['dvs']['pol']  # Event polarity (1 for ON events, 0 for OFF events)

# Define the dimensions of the event-based camera (304 pixels width by 240 pixels height)
width = 304
height = 240

###############################
####### create networks #######
###############################

# Parameters OMS
size_krn_center = 8  # Size of the kernel (NxN) (all half ) - 8
sigma_center = 1  # Sigma for the first Gaussian - 1
size_krn_surround = 8  # Size of the kernel (NxN) - 8
sigma_surround = 4  # Sigma for the first Gaussian - 4

# Parameters OMS
threshold = 0.80
num_pyr = 1


# Parameters visual attention
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



device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

##### egomotion #####
center, surround = OMSkernels(size_krn_center, sigma_center, size_krn_surround, sigma_surround)
ss = 1
sc = ss + sigma_surround - sigma_center
net_center = net_def(center, tau_mem, num_pyr, size_krn_center, device, sc)
net_surround = net_def(surround, tau_mem, num_pyr, size_krn_surround, device, ss)


##### attention #####
VMkernels = create_vm_filters(thetas, size, rho, r0, thick, offset, fltr_resize_perc)
# plot_filters(filters_attention, thetas)
netattention = net_def(VMkernels, tau_mem, num_pyr, size, device, 1)


###### Define window  visualisation #######
# Define the window period for event visualization (in milliseconds)
time_window=200  #  ms window for showing events
window_period=200  #  ms window for showing events

# Create an empty window (image) to store event data for each time window
window = np.zeros((height, width))

# Loop through all the events (x, y, timestamp, polarity)
for x, y, ts, pol in zip(e_x, e_y, e_ts, e_pol):

    # Check if the event timestamp is within the current time window (less than or equal to window_period)
    if ts <= window_period:
        # If within the window, set the pixel at (y, x) to 255 (bright pixel) to visualize the event
        window[y][x] = 255
    else:
        # If the event is outside the window period, plot the current window of events
        plt.imshow(window)  # Display the image showing the events
        plt.draw()  # Update the figure with the new window of events
        plt.pause(0.2)  # Pause for 0.2 seconds to allow the plot to be displayed interactively
        # Update the window period by increasing it by the window period (1 ms) for the next set of events
        window_period += time_window
        # Reset the window to zero for the next batch of events
        window = np.zeros((height, width))

# Print 'end' to indicate the processing and visualization of events is complete
print('end')
