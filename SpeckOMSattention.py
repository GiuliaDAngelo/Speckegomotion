from functions.Speck_funcs import *
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


if __name__ == "__main__":

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
                    egomap, indexes = egomotion(window, netegomotion, numevs, device)
                    salmap, salmap_coords = run_attention(egomap, netattention, device)
                    if showstats:
                        attention_maps.append(salmap)
                        salmaps_coordinates.append(salmap_coords)
                        windows.append(egomap[0])
                    cv2.imshow('Events', window)
                    cv2.imshow('Events after Suppression', egomap[0])
                    cv2.circle(salmap, (salmap_coords[1], salmap_coords[0]), 5, (255, 255, 255), -1)
                    cv2.imshow('Saliency Map', cv2.applyColorMap(cv2.convertScaleAbs(salmap), cv2.COLORMAP_JET))
                    cv2.waitKey(1)
                    window.fill(0)
                    if pantiltunit:
                        # are the location of maximum attention value, values are between -1 and 1
                        cmd = run_controller(
                                np.array([salmap_coords[1]/(resolution[1]), salmap_coords[0]/(resolution[1])]),
                                np.array([0.5, 0.5]),
                                k_pan=np.array([2.,0.,0.]),
                                k_tilt=np.array([2.,0.,0.]),
                            )

                        # sending commands to the pantilt unit; format the command (assuming the device uses "PP<angle>" and "TP<angle>")
                        # rescale cmd to pan_range & tilt_range
                        pan_angle = int((cmd[0] * (pan_range[1] - pan_range[0]) / 2) + (pan_range[1] + pan_range[0]) / 2)
                        tilt_angle = int(
                            (cmd[1] * (tilt_range[1] - tilt_range[0]) / 2) + (tilt_range[1] + tilt_range[0]) / 2)

                        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
                            pan_command = f'PP{pan_angle}\n'
                            tilt_command = f'TP{tilt_angle}\n'

                            # Send the pan and tilt commands
                            send_command(ser, f'PU\n')
                            send_command(ser, pan_command)
                            send_command(ser, f'TU\n')
                            send_command(ser, tilt_command)
                            send_command(ser, f'A\n')
                            response = ser.readline().decode('utf-8').strip()
                        if response:
                            print(f"Response from device: {response}")
                            pass
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
