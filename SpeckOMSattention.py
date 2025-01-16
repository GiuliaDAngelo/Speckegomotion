from functions.Speck_helpers import *
from functions.attention_helpers import *
from functions.OMS_helpers import *
from controller.helper import run_controller
import serial
import matplotlib
import cv2
import time
import numpy as np
matplotlib.use('TkAgg')



# Visualization parameters
resolution = [128, 128] # Resolution of the DVS sensor
max_x = resolution[0]
max_y = resolution[1]
drop_rate = 0.0  # Percentage of events to drop
update_interval = 0.001 #0.02 #seconds
last_update_time = time.time()


# Parameters OMS
size_krn_center = 8  # Size of the kernel (NxN) (all half ) - 8
sigma_center = 1  # Sigma for the first Gaussian - 1
size_krn_surround = 8  # Size of the kernel (NxN) - 8
sigma_surround = 4  # Sigma for the first Gaussian - 4

# Parameters network
threshold = 0.80
num_pyr = 1


# Parameters attention
size = 10  # Size of the kernel
r0 = 4  # Radius shift from the center
rho = 0.1  # Scale coefficient to control arc length
theta = np.pi * 3 / 2  # Angle to control the orientation of the arc
thick = 3  # thickness of the arc
offsetpxs = 0 #size / 2
offset = (offsetpxs, offsetpxs)
fltr_resize_perc = [2, 2]
num_pyr = 3
# The angles are generated in radians, ranging from 0 to 2π in steps of π/4
thetas = np.arange(0, 2 * np.pi, np.pi / 4)



tau_mem = 0.1
if __name__ == "__main__":

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    ##### egomotion #####
    center, surround = OMSkernels(size_krn_center, sigma_center, size_krn_surround, sigma_surround)
    ss = 1
    sc = ss + sigma_surround - sigma_center
    net_center = net_def(center, tau_mem, num_pyr, size_krn_center, device, sc)
    net_surround = net_def(surround, tau_mem, num_pyr, size_krn_surround, device, ss)

    ##### attention #####
    VMkernels = create_vm_filters(thetas, size, rho, r0, thick, offset)
    # plot_filters(filters_attention, thetas)
    netattention = net_def(VMkernels, tau_mem, num_pyr, size, device, 1)

    ##### Speck initialisation for the sink #####
    # Set the Speck and sink events
    sink, window, numevs, events_lock = Specksetup()
    # Start the event-fetching thread
    event_thread = threading.Thread(target=fetch_events, args=(sink, window, drop_rate, events_lock, numevs))
    event_thread.daemon = True
    event_thread.start()
    showstats = False
    pantiltunit = True
    if showstats:
        plt.figure()
        num_eve = []
        num_eve_supp = []
        num_eve_drop = []
        attention_maps = []
        salmaps_coordinates = []
        windows = []
    # Main loop for visualization
    start_time = time.time()
    last_update_time = 0

    while True:
        current_time = time.time()-start_time
        with events_lock:
            if current_time - last_update_time > update_interval:
                if numevs[0] > 0:
                    egomap, indexes = egomotion(window, net_center, net_surround, device, max_y, max_x, threshold)
                    salmap, salmap_coords = run_attention(egomap, netattention, device)
                    if showstats:
                        attention_maps.append(salmap)
                        salmaps_coordinates.append(salmap_coords)
                        windows.append(egomap[0])
                    cv2.imshow('Events map', window)
                    cv2.imshow('OMS map', egomap[0])
                    cv2.circle(salmap, (salmap_coords[1], salmap_coords[0]), 5, (255, 255, 255), -1)
                    cv2.imshow('Saliency Map', cv2.applyColorMap(cv2.convertScaleAbs(salmap), cv2.COLORMAP_JET))
                    cv2.waitKey(1)
                    window.fill(0)
                    if showstats:
                        #print number of events
                        # print('Number of events: ' + str(numevs[0]))
                        # print('Number of suprressed events:', indexes.sum().item())
                        num_eve.append(numevs[0])
                        num_eve_supp.append(indexes.sum().item())
                        num_eve_drop.append(numevs[0] - indexes.sum().item())
                        plt.plot([current_time], [numevs[0]], 'ro-', label='Events')
                        plt.plot([current_time], [indexes.sum().item()], 'bo-', label='Events after suppression')
                        plt.plot([current_time], [numevs[0] - indexes.sum().item()], 'yo-', label='Events dropping')
                        # plt.title('Comparison of Events before and after suppression')
                        plt.xlabel('Time [s]')
                        plt.ylabel('Events Count')
                        if not plt.gca().get_legend():
                            plt.legend()
                        plt.pause(0.001)  # Pause to update the figure
                        # current time > 30 save figure and break
                        if current_time > 10:
                            plt.savefig('events_comparison.png')
                            # save num_evets, num_evets_supp, num_evets_drop
                            print(np.mean(num_eve))
                            print(np.mean(num_eve_supp))
                            print(np.mean(num_eve_drop))
                            #save attention maps and salmaps_coordinates
                            np.save('attention_maps.npy', attention_maps)
                            np.save('salmaps_coordinates.npy', salmaps_coordinates)
                            np.save('windows.npy', windows)
                            break
                    numevs[0] = 0
                last_update_time = current_time
