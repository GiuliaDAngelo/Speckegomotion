# --------------------------------------
import numpy as np

from functions.Speck_helpers import *

# --------------------------------------
from controller.helper import run_controller

# --------------------------------------
import serial

# --------------------------------------
from datetime import datetime

# --------------------------------------
from loguru import logger

# --------------------------------------
from dataclasses import dataclass

# --------------------------------------
from multiprocessing import Event

# --------------------------------------
from functions.OMS_helpers import *

# --------------------------------------
from functions.attention_simple_helpers import *

import matplotlib

# tkagg
matplotlib.use('TkAgg')

global microsaccades, rnd, alpha, serial_port, baud_rate, ser, dxs, dys, meanxs, meanys, latency

@dataclass
class Flags:
    events = Event()
    pantilt = Event()
    movement = Event()
    halt = Event()

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
        'sc': 1,
        'ss': 1
    }

    # Attention Parameters
    ATTENTION_PARAMS = {
        'size_krn' : 10 , # Size of the kernel
        'r0' : 4 , # Radius shift from the center
        'rho' : 0.1 , # Scale coefficient to control arc length
        'theta' : np.pi * 3 / 2, # Angle to control the orientation of the arc
        'thetas' : np.arange(0, 2 * np.pi, np.pi / 4),
        'thick' : 3, # thickness of the arc
        'offsetpxs' : 0, # size / 2
        'offset' : (offsetpxs, offsetpxs),
        'fltr_resize_perc' : [2, 2],
        'num_pyr' : 3,
        'tau_mem': 0.3,
        'stride':1,
        'out_ch':1
    }

flags = Flags()
config = Config()

matplotlib.use("TkAgg")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

def perform_attention_with_pan_tilt(
        vSliceOMS: torch.Tensor,
        saliency_map: np.ndarray,
        OMS: np.ndarray,
        salmax_coords: np.ndarray,
        cmd: np.ndarray,
        counter: int,
        flags: Flags = None,
        pan_range: np.ndarray = None,
        tilt_range: np.ndarray = None,
):

    while not flags.halt.is_set():
        # time.sleep(0.2)
        # compute attention
        saliency_map[:], salmax_coords[:] = run_attention(vSliceOMS, net_attention, device, resolution,
                                                          size_krn_after_oms, num_pyr=3)

        # cmds values are between -1 and 1
        cmd[:] = run_controller(
            np.array([salmax_coords[0], salmax_coords[1]]),
            np.array([resolution[1] // 2, resolution[0] // 2]),
            k_pan=np.array([alpha, 0., 0.]),
            k_tilt=np.array([alpha, 0., 0.]),
        )

        # delta pan and tilt to obtain an output between -1 and 1
        # convert the -1, 1 to actual degrees of pan and tilt
        delta_pan = int(cmd[1] / degrees_per_pos / alpha / 2)
        delta_tilt = int(cmd[0] / degrees_per_pos / alpha / 2)

        # make a check if pan_angle and tilt_angle are within the range
        pan_angle = np.clip(delta_pan, pan_range[0], pan_range[1]).astype(int)
        tilt_angle = np.clip(delta_tilt, tilt_range[0], tilt_range[1]).astype(int)

        logger.info(
            f"Moving | pan (salmax, cmd, delta): {salmax_coords[1], cmd[1], delta_pan} | tilt (salmax, cmd, delta): {salmax_coords[0], cmd[0], delta_tilt} | pan_angle: {pan_angle} / {pan_range} | tilt_angle {tilt_angle} / {tilt_range}")

        # Send the pan and tilt commands
        send_command(ser, f'PS1000\n') #speed
        pan_command = f'PP{pan_angle}\n'
        send_command(ser, f'PS1000\n') #speed
        tilt_command = f'TP{tilt_angle}\n'
        # Send the pan and tilt commands
        send_command(ser, pan_command)
        send_command(ser, tilt_command)
        send_command(ser, f'A\n')

        for i in range(microsaccades):
            send_command(ser, f'PP{pan_angle + rnd[i]}\n')
            send_command(ser, f'TP{tilt_angle + rnd[i]}\n')
        send_command(ser, f'A\n')
        if response:
            print(f"Response from device: {response}")


def clean_up():
    flags.halt.set()
    flags.events.set()
    flags.pantilt.set()

def dvs_events_to_numpy(events):
    return np.array([(ev.x, ev.y, ev.p, ev.timestamp) for ev in events if isinstance(ev, samna.speck2f.event.DvsEvent)],
                    dtype=[('x', 'u1'), ('y', 'u1'), ('p', bool), ('t', 'u4')], )


def visualiser(dxs, dys):
    combined = np.full((280, 280, 3), fill_value=255, dtype=np.uint8)
    combined[1:129, 1:129, :] = window[:, :, None].repeat(3, axis=2) #(1r1c) - events
    combined[-121:, 1:122, :] = saliency_map[:, :, None].repeat(3, axis=2) #(2r1c) - saliency
    combined[1:122, -121:, :]  = OMS[:, :, None].repeat(3, axis=2)

    black_wind = np.zeros_like(saliency_map)
    black_wind = np.stack([black_wind] * 3, axis=2)
    cv2.circle(black_wind, (salmax_coords[1], salmax_coords[0]), 6, (0, 1, 0), -1)
    cv2.circle(black_wind, (64, 64), 3, (0, 0, 1), -1)
    cv2.circle(black_wind, (64 - cmd[1], 64 - cmd[0]), 3, (0, 1, 1), -1)
    dxs.append(abs(salmax_coords[0] - (64 - cmd[0])))
    dys.append( abs(salmax_coords[1] - (64 - cmd[1])))
    combined[-121:, -121:, :] = black_wind * 255 #(2r2c) - control

    cv2.putText(combined, 'Events map', (10, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    cv2.putText(combined, 'Saliency map', (8, 270), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    cv2.putText(combined, 'sOMS map', (175, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    cv2.putText(combined, 'sFC coord', (170, 270), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    cv2.imshow('', combined)
    cv2.waitKey(1)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    microsaccades = 14
    rnd = np.random.uniform(-60, 60, microsaccades).astype(np.int32)
    alpha = 3.1
    # Define the serial port and settings
    serial_port = "/dev/tty.usbserial-FTGZO55F"
    baud_rate = 9600
    ser = None
    if ser is None:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
    dxs, dys = [], []
    meanxs, meanys = [], []
    latency = []

    ##### egomotion #####
    # Initialize OMS
    net_center, net_surround = initialize_oms(device, config.OMS_PARAMS)

    ##### attention #####
    # Initialize Attention modules
    net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

    # Define the pan and tilt ranges
    degrees_per_pos = 0.02572  # one position (92.5714 seconds arc) is 92.5714 sec arc * 0.0002778° = 0.02572°
    focal_length = 1.7  # mm
    sensor_size_hor = 5.12  # mm 1/2.5" sensor size - A 1/2.5" sensor typically has a diagonal size of approximately 5.76 mm and a horizontal size of about 4.6 mm.

    field_of_view = 2 * np.arctan(sensor_size_hor / (2 * focal_length))
    afield_of_view = np.degrees(field_of_view)
    ptrange = int(afield_of_view / degrees_per_pos) // 2

    pan_range_vision_sensor = np.array([-ptrange, ptrange])
    tilt_range_vision_sensor = np.array([-ptrange, ptrange])

    pan_range = np.clip(pan_range_vision_sensor, -3090, 3090)
    tilt_range = np.clip(tilt_range_vision_sensor, -907, 604)

    pan_angle = 0
    tilt_angle = 0

    # Set first position to 0
    # with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
    pan_command = f'PP{0}\n'
    tilt_command = f'TP{0}\n'
    # Send the pan and tilt commands
    send_command(ser, pan_command)
    send_command(ser, tilt_command)
    send_command(ser, f'A\n')
    response = ser.readline().decode('utf-8').strip()

    ##### Speck initialisation for the sink #####
    # Set the Speck and sink events
    sink, window, window_pos, window_neg, numevs, events_lock = Specksetup()

    # Initialize the models output
    size_krn_after_oms = 121
    OMS = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
    vSliceOMS = torch.zeros((1, size_krn_after_oms, size_krn_after_oms), dtype=torch.float32)
    saliency_map = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
    salmax_coords = np.zeros((2,), dtype=np.int32)
    cmd = np.zeros((2,), dtype=np.int32)
    counter = 0
    trials = 5300
    end_trials = trials * 10

    # Starting attention thread
    attention_thread = threading.Thread(
        target=perform_attention_with_pan_tilt,
        args=(vSliceOMS, saliency_map, OMS, salmax_coords, cmd, counter, flags, pan_range, tilt_range),
    )
    attention_thread.daemon = True
    attention_thread.start()
    salmap_coords = np.array([0, 0])

    flags.events.set()
    while not flags.halt.is_set():
        current_time = datetime.now()
        logger.info("Processing...")
        try:
            while True:
                logger.info("Waiting for attention to finish...")
                if flags.halt.is_set():
                    break
                start_time = time.time()
                events = sink.get_events_blocking(1000)  # ms
                counter += 1
                if counter>=trials:
                    print('dys mean: {:.2f}'.format(np.mean(dys)), 'dxs mean: {:.2f}'.format(np.mean(dxs)))
                    print('dys std: {:.2f}'.format(np.std(dys)), 'dxs std: {:.2f}'.format(np.std(dxs)))
                    meanxs.append(np.mean(dxs))
                    meanys.append(np.mean(dys))
                    dxs, dys = [], []
                    trials += counter
                if counter >= end_trials:
                    print('All trials dys mean: {:.2f}'.format(np.mean(meanys)), 'All trials dxs mean: {:.2f}'.format(np.mean(meanxs)))
                    print('All trials dys std: {:.2f}'.format(np.std(meanys)), 'All trials dxs std: {:.2f}'.format(np.std(meanxs)))
                    print('Mean latency: {:.2f}'.format(np.std(latency)),
                          'Std latency: {:.2f}'.format(np.std(latency)))
                    pan_command = f'PP{0}\n'
                    tilt_command = f'TP{0}\n'
                    # Send the pan and tilt commands
                    send_command(ser, pan_command)
                    send_command(ser, tilt_command)
                    send_command(ser, f'A\n')
                    response = ser.readline().decode('utf-8').strip()
                    break
                if events:
                    with events_lock:
                        window[:] = 0
                        curr_events = dvs_events_to_numpy(events)
                        # extract polarities
                        window[curr_events['x'], curr_events['y']] = 255
                        window[:] = np.rot90(window, k=1)
                        # ###### computing OMS and attention ######
                        wOMS = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

                        # computing egomotion
                        OMS, indexes = egomotion(wOMS, net_center, net_surround, device, config.MAX_Y,
                                                 config.MAX_X, config.OMS_PARAMS['threshold'])
                        # prepare vSliceOMS
                        vSliceOMS[:] = OMS.squeeze(0)
                        OMS = OMS.squeeze(0).squeeze(0).cpu().detach().numpy()
                        numevs[0] += len(events)
                        visualiser(dxs, dys)
                        latency.append(time.time() - start_time)
                else:
                    logger.info("No events, exiting...")
                    flags.halt.set()

        except KeyboardInterrupt:
            flags.halt.set()
            break

        finally:
            clean_up()



