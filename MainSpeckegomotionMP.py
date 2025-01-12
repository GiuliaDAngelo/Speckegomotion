# --------------------------------------
import numpy as np
import torch
from torch.fx.experimental.proxy_tensor import set_proxy_slot

from Speckegolayer_functions import *

# --------------------------------------
from configSpeckmain import *

# --------------------------------------
from controller.helper import run_controller

# --------------------------------------
import serial

# --------------------------------------
import matplotlib

# --------------------------------------
from datetime import datetime

# --------------------------------------
from multiprocessing import Process

# # --------------------------------------
# import pyqtgraph as pg

# --------------------------------------
from loguru import logger

# --------------------------------------
from dataclasses import dataclass

# --------------------------------------
from multiprocessing import Event

# --------------------------------------
from functions.OMS_helpers import *

# --------------------------------------
from functions.attention_helpers import *

import matplotlib
#tkagg
matplotlib.use('TkAgg')

@dataclass
class Flags:
    events = Event()
    pantilt = Event()
    movement = Event()
    halt = Event()


matplotlib.use("TkAgg")
fig, ax = plt.subplots(1,1, figsize=(12,8))


def perform_attention_with_pan_tilt(
    vSliceOMS: torch.Tensor,
    saliency_map: np.ndarray,
    salmax_coords: np.ndarray,
    cmd: np.ndarray,
    counter: int,
    dummy_pan_tilt: bool = False,
    flags: Flags = None,
    pan_range: np.ndarray = None,
    tilt_range: np.ndarray= None,
):

    while not flags.halt.is_set():

        with torch.no_grad():
            saliency_map[:] = net_attention(vSliceOMS).cpu().numpy()
            salmax_coords[:] = np.unravel_index(torch.argmax(torch.tensor(saliency_map)).cpu().numpy(), saliency_map.shape)

        # #tilt
        # if (counter % 20)  == 0:
        #     if salmax_coords[0] == 0:
        #         salmax_coords[:] = (120,64)
        #     else:
        #         salmax_coords[:] = (0,64)
        #
        # #pan
        # if (counter % 20) == 0:
        #     if salmax_coords[1] == 0:
        #         salmax_coords[:] = (64, 120)
        #     else:
        #         salmax_coords[:] = (64, 0)

    # logger.info("Waiting for events to accumulate...")
    #     flags.pantilt.wait()

        # if flags.halt.is_set():
        #     break

        # cmds values are between -1 and 1
        alpha=1.
        cmd[:] = run_controller(
                np.array([salmax_coords[0], salmax_coords[1]]),
                np.array([resolution[1] // 2, resolution[0] // 2]),
                k_pan=np.array([alpha, 0., 0.]),
                k_tilt=np.array([alpha, 0., 0.]),
            )


        # delta pan and tilt to obtain an output between -1 and 1
        # convert the -1, 1 to actual degrees of pan and tilt
        delta_pan = int(cmd[1] // degrees_per_pos)
        delta_tilt =  int(cmd[0]// degrees_per_pos)

        #make a check if pan_angle and tilt_angle are within the range
        pan_angle = np.clip(delta_pan, pan_range[0], pan_range[1]).astype(int)
        tilt_angle = np.clip(delta_tilt, tilt_range[0], tilt_range[1]).astype(int)

        logger.info(f"Moving | pan (salmax, cmd, delta): {salmax_coords[1], cmd[1], delta_pan} | tilt (salmax, cmd, delta): {salmax_coords[0], cmd[0], delta_tilt} | pan_angle: {pan_angle} / {pan_range} | tilt_angle {tilt_angle} / {tilt_range}")

        # Dummy pan & tilt
        if dummy_pan_tilt:
            # time.sleep(0.5)
            pass
        else:
            with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
                pan_command = f'PP{pan_angle}\n'
                tilt_command = f'TP{tilt_angle}\n'
                # Send the pan and tilt commands
                # send_command(ser, f'PU\n')
                send_command(ser, pan_command)
                # send_command(ser, f'TU\n')
                send_command(ser, tilt_command)
                send_command(ser, f'A\n')
                response = ser.readline().decode('utf-8').strip()
                # time.sleep(0.1)
            if response:
                print(f"Response from device: {response}")
        #
        # # logger.info("Attention complete")
        # flags.pantilt.clear()


def clean_up():    
    flags.halt.set()
    flags.events.set()
    flags.pantilt.set()

def dvs_events_to_numpy(events):
    return np.array([(ev.x, ev.y, ev.p, ev.timestamp)for ev in events if isinstance(ev, samna.speck2f.event.DvsEvent)],dtype = [('x', 'u1'), ('y', 'u1'), ('p', bool), ('t', 'u4')], )

class Config:
    MAX_X, MAX_Y = 128, 128
    # OMS Parameters
    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.86,
        'tau_memOMS': 0.3,
        'sc':1,
        'ss':1
    }

    # Attention Parameters
    ATTENTION_PARAMS = {
        'VM_radius': 8, #(R0)
        'VM_radius_group': 15,
        'num_ori': 4,
        'b_inh': 3, #(w)
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2, #(rho)
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.3
    }


if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    flags = Flags()
    config = Config()

    ##### egomotion #####
    # Initialize OMS
    net_center, net_surround = initialize_oms(device, config.OMS_PARAMS)

    ##### attention #####
    # Initialize Attention modules
    net_attention = AttentionModule(**config.ATTENTION_PARAMS)

    #### PanTilt setup ####

    # Define the serial port and settings
    serial_port = "/dev/tty.usbserial-FTGZO55F"
    baud_rate = 9600

    #### pan-tilt range pan: -3090/3090 - tilt:-907 and 604

    degrees_per_pos = 0.02572
    # pan_fov_range = 30 # degrees as per humans
    # degrees_per_pixels = pan_fov_range / 128
    # # steps_per_pixel = degrees_per_pixels / degrees_per_pos
    # # to obtain the positions for visual field of teh camera
    #
    tilt_fov_range_up = 10 # degrees as per humans (should be 20)
    tilt_fov_range_down = 10  # degrees as per humans (should be 47)

    # Define pan and tilt range values
    # pan_range = np.array([
    #     -((pan_fov_range/2) // degrees_per_pos),
    #     ((pan_fov_range/2) // degrees_per_pos),
    # ])  # Replace with actual pan range of your device if different
    pan_range = np.array([-583,583])
    # tilt_range = np.array([
    #     -(tilt_fov_range_down // degrees_per_pos),
    #     (tilt_fov_range_up // degrees_per_pos),
    # ])  # Replace with actual tilt range of your device if different
    tilt_range = np.array([-388, 388])

    with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
        pan_command = f'PP{0}\n'
        tilt_command = f'TP{0}\n'
        # Send the pan and tilt commands
        # send_command(ser, f'PU\n')
        send_command(ser, pan_command)
        # send_command(ser, f'TU\n')
        send_command(ser, tilt_command)
        send_command(ser, f'A\n')
        response = ser.readline().decode('utf-8').strip()

    ##### Speck initialisation for the sink #####
    # Set the Speck and sink events
    sink, window_pos, window_neg, numevs, events_lock = Specksetup()

    # A switch for emulating pan/tilt motion
    showstats = False
    pantiltunit = True
    dummy_pan_tilt = False

    vSliceOMS = torch.zeros((1,2,121,121), dtype = torch.float32).to(device)
    saliency_map = np.zeros((121, 121), dtype = np.float32)
    salmax_coords = np.zeros((2,), dtype = np.int32)
    cmd = np.zeros((2,), dtype = np.int32)
    counter = 0

    attention_thread = threading.Thread(
        target=perform_attention_with_pan_tilt,
        args=(vSliceOMS, saliency_map, salmax_coords, cmd, counter, dummy_pan_tilt, flags, pan_range, tilt_range),
    )
    attention_thread.daemon = True
    attention_thread.start()

    salmap_coords = np.array([0, 0])


    if showstats:
        plt.figure()
        num_eve = []
        num_eve_supp = []
        num_eve_drop = []
        attention_maps = []
        salmaps_coordinates = []
        windows = []
    # Main loop for visualization

    flags.events.set()
    while not flags.halt.is_set():
        current_time = datetime.now()
        logger.info("Processing...")

        try:

            while True:
                # logger.info("Waiting for attention to finish..."
                if flags.halt.is_set():
                    break
                # logger.info("Accumulating events...")
                # time.sleep(random.gauss(1, 0.05))
                
                events = sink.get_events_blocking(2000)  # ms
                counter += 1
                if events:
                    with events_lock:
                        window_pos[:] = 0
                        window_neg[:] = 0
                        curr_events = dvs_events_to_numpy(events)
                        # extract polarities
                        pos_ev = curr_events[curr_events['p'] == 1]
                        neg_ev = curr_events[curr_events['p'] == 0]

                        window_pos[pos_ev['x'], pos_ev['y']] = 255
                        window_neg[neg_ev['x'], neg_ev['y']] = 255

                        window_pos[:] = np.rot90(window_pos, k=1)
                        window_neg[:] = np.rot90(window_neg, k=1)
                        numevs[0] += len(events)

                    # ###### computing OMS and attention ######
                    # visualisation window
                    window_combined = window_pos + window_neg
                    #
                    # OMS (split in polarities)
                    wOMS_pos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(device)
                    wOMS_neg = torch.tensor(window_neg, dtype=torch.float32).unsqueeze(0).to(device)

                    # computing egomotion
                    OMSpos, indexes_pos = egomotion(wOMS_pos, net_center, net_surround, device, config.MAX_Y,
                                                    config.MAX_X, config.OMS_PARAMS['threshold'])
                    OMSneg, indexes_neg = egomotion(wOMS_neg, net_center, net_surround, device, config.MAX_Y,
                                                    config.MAX_X, config.OMS_PARAMS['threshold'])
                    # prepare vSliceOMS
                    OMSpos = OMSpos.squeeze(0).squeeze(0).to(device)
                    OMSneg = OMSneg.squeeze(0).squeeze(0).to(device)
                    vSliceOMS[:] = torch.stack((OMSpos,OMSneg)).unsqueeze(0).float()
                    #
                    # visualisation of OMS
                    OMS = OMSpos + OMSneg
                    OMS[OMS != 0] = 1.0 * 255



                    ## need to pass it to a low pass filter, running mean and use the running mean to compute the pan tilt

                    # flags.pantilt.set()

                    cv2.imshow('Events map', window_combined)
                    cv2.imshow('OMS map', OMS.cpu().numpy())
                    black_wind = np.zeros_like(saliency_map)
                    black_wind = np.stack([black_wind] * 3, axis = 2)
                    cv2.circle(black_wind, (salmax_coords[1], salmax_coords[0]), 6, (0, 1, 0), -1)
                    cv2.circle(black_wind, (64 - cmd[1], 64 - cmd[0]), 3, (0, 0, 1), -1)
                    cv2.imshow('Cmd Map', black_wind)
                    cv2.imshow('Saliency Map', saliency_map)
                    cv2.waitKey(1)
                
                else:
                    logger.info("No events, exiting...")
                    flags.halt.set()

        except KeyboardInterrupt:
            flags.halt.set()
            break

        finally:
            clean_up()
            # attention_thread.join()
