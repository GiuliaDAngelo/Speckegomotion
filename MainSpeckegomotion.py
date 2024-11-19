from Speckegolayer_functions import *
from configSpeckmain import *
from controller.helper import run_controller
import serial
import matplotlib
from datetime import datetime
from dataclasses import dataclass
from threading import Event
from threading import Thread
matplotlib.use('TkAgg')
from loguru import logger


@dataclass
class Flags:
    attention = Event()
    halt = Event()

flags = Flags()

def perform_attention_with_pan_tilt(dummy_pan_tilt: bool = True, showstats: bool = False):

    while not flags.halt.is_set():

        logger.info("Waiting for attention to start")
        flags.attention.wait()
        logger.info("Attention started")

        if flags.halt.is_set():
            break

        logger.info("Computing egomotion")
        egomap, indexes = egomotion(window, netegomotion, numevs, device)

        logger.info("Computing attention")
        salmap, salmap_coords = run_attention(egomap, netattention, device)
        # are the location of maximum attention value, values are between -1 and 1
        cmd = run_controller(
                np.array([salmap_coords[1]/(resolution[1]), salmap_coords[0]/(resolution[1])]),
                np.array([0.5, 0.5])
            )

        pan_angle = int((cmd[0] * (pan_range[1] - pan_range[0]) / 2) + (pan_range[1] + pan_range[0]) / 2)
        tilt_angle = int(
            (cmd[1] * (tilt_range[1] - tilt_range[0]) / 2) + (tilt_range[1] + tilt_range[0]) / 2)

        # cv2.imshow('Events', window)
        # cv2.imshow('Events after Suppression', egomap[0])
        # cv2.circle(salmap, (salmap_coords[1], salmap_coords[0]), 5, (255, 255, 255), -1)
        # cv2.imshow('Saliency Map', cv2.applyColorMap(cv2.convertScaleAbs(salmap), cv2.COLORMAP_JET))
        # cv2.waitKey(1)

        if showstats:
            #print number of events
            # print('Number of events: ' + str(numevs[0]))
            # print('Number of suprressed events:', indexes.sum().item())
            plt.plot([current_time], [numevs[0]], 'ro-', label='Events')
            plt.plot([current_time], [indexes.sum().item()], 'bo-', label='Events after suppression')
            plt.plot([current_time], [numevs[0] - indexes.sum().item()], 'yo-', label='Events dropping')
            plt.title('Comparison of Events before and after suppression')
            plt.xlabel('Time')
            plt.ylabel('Events Count')
            if not plt.gca().get_legend():
                plt.legend()
            plt.pause(0.001)  # Pause to update the figure

        # Dummy pan & tilt
        if dummy_pan_tilt:
            time.sleep(0.5)
        else:
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

        logger.info("Attention complete")
        flags.attention.clear()


def clean_up():
    flags.halt.set()
    flags.attention.set()

if __name__ == "__main__":


    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    ##### egomotion #####

    # loading egomotion kernel
    filter_egomotion = egokernel()

    # Initialize the network with the loaded filter
    netegomotion = net_def(filter_egomotion,tau_mem, num_pyr, filter_egomotion.size(1))

    ##### attention #####

    # loading attention kernels
    filters_attention = create_vm_filters(thetas, size, rho, r0, thick, offset)
    # plot_filters(filters_attention, thetas)

    # Initialize the attention network with the loaded filters
    netattention = network_init(filters_attention)

    #### PanTilt setup ####

    # Define the serial port and settings
    serial_port = '/dev/tty.usbserial-FTGZO55F'
    baud_rate = 9600

    # Define pan and tilt range values
    pan_range = (-3090, 3090)  # Replace with actual pan range of your device if different
    tilt_range = (-907, 604)  # Replace with actual tilt range of your device if different

    ##### Speck initialisation for the sink #####
    # Set the Speck and sink events
    sink, window, numevs, events_lock = Specksetup()

    # A switch for emulating pan/tilt motion
    showstats = False
    pantiltunit = True
    dummy_pan_tilt = True

    # Start the event-fetching thread
    event_thread = Thread(target=fetch_events, args=(sink, window, drop_rate, events_lock, numevs))
    event_thread.daemon=True
    event_thread.start()

    attention_thread = Thread(target=perform_attention_with_pan_tilt, args=(dummy_pan_tilt, showstats), daemon=True)
    attention_thread.daemon=True
    attention_thread.start()

    if showstats:
        plt.figure()
        num_eve = []
        num_eve_supp = []
        num_eve_drop = []
        attention_maps = []
        salmaps_coordinates = []
        windows = []
    # Main loop for visualization
    while True:
        current_time = time.time()
        with events_lock:
            if current_time - last_update_time > update_interval:
                if numevs[0] > 0:
                    egomap, indexes = egomotion(window, netegomotion, numevs, device)
                    salmap, salmap_coords = run_attention(egomap, netattention, device)
                    # are the location of maximum attention value, values are between -1 and 1
                    # cmd = run_controller(
                    #         np.array([salmap_coords[1]/(resolution[1]), salmap_coords[0]/(resolution[1])]),
                    #         np.array([0.5, 0.5])
                    #     )
                    cv2.imshow('Events', window)
                    cv2.imshow('Events after Suppression', egomap[0])
                    cv2.circle(salmap, (salmap_coords[1], salmap_coords[0]), 5, (255, 255, 255), -1)
                    cv2.imshow('Saliency Map', cv2.applyColorMap(cv2.convertScaleAbs(salmap), cv2.COLORMAP_JET))
                    cv2.waitKey(1)
                    window.fill(0)
                    # sending commands to the pantilt unit; format the command (assuming the device uses "PP<angle>" and "TP<angle>")
                    # rescale cmd to pan_range & tilt_range
                    # pan_angle = int((cmd[0] * (pan_range[1] - pan_range[0]) / 2) + (pan_range[1] + pan_range[0]) / 2)
                    # tilt_angle = int(
                    #     (cmd[1] * (tilt_range[1] - tilt_range[0]) / 2) + (tilt_range[1] + tilt_range[0]) / 2)
                    if pantiltunit:
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
                        plt.plot([current_time], [numevs[0]], 'ro-', label='Events')
                        plt.plot([current_time], [indexes.sum().item()], 'bo-', label='Events after suppression')
                        plt.plot([current_time], [numevs[0] - indexes.sum().item()], 'yo-', label='Events dropping')
                        plt.title('Comparison of Events before and after suppression')
                        plt.xlabel('Time')
                        plt.ylabel('Events Count')
                        if not plt.gca().get_legend():
                            plt.legend()
                        plt.pause(0.001)  # Pause to update the figure
                    numevs[0] = 0
                last_update_time = current_time

        finally:
            clean_up()
            event_thread.join()
            attention_thread.join()