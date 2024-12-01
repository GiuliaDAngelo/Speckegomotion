# --------------------------------------
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

# --------------------------------------
import pyqtgraph as pg

# --------------------------------------
from loguru import logger

# --------------------------------------
from dataclasses import dataclass

# --------------------------------------
from multiprocessing import Event


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
    dummy_pan_tilt: bool = True,
    showstats: bool = False,
    flags: Flags = None,
    suppression: np.ndarray = None,
    pr: np.ndarray = None,
    tr: np.ndarray= None,
):

    pan_norm = (pr[1] - pr[0]) / 2
    tilt_norm = (tr[1] - tr[0]) / 2

    while not flags.halt.is_set():

        # logger.info("Waiting for events to accumulate...")
        flags.pantilt.wait()

        if flags.halt.is_set():
            break

        # cmds values are between -1 and 1
        cmd = run_controller(
                np.array([salmap_coords[1]/(resolution[1]), salmap_coords[0]/(resolution[0])]),
                np.array([0.5, 0.5]),
                k_pan=np.array([2., 0., 0.]),
                k_tilt=np.array([2., 0., 0.]),
            )

        alpha = 10
        pan_angle = (pr[0] + (cmd[0]*alpha + 1) * pan_norm).astype(np.int32)
        tilt_angle =(tr[0] + (cmd[1]*alpha + 1) * tilt_norm).astype(np.int32)
        #make a check if pan_angle and tilt_angle are within the range
        pan_angle = np.clip(pan_angle, pan_range[0], pan_range[1])
        tilt_angle = np.clip(tilt_angle, tilt_range[0], tilt_range[1])

        logger.info(f"Moving | cmd: ({cmd[0]:>0.3f},{cmd[1]:>0.3f}) | salmap coords:  {salmap_coords} | pan_angle: {pan_angle} / {pr} | tilt_angle {tilt_angle} / {tilt_range}")

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

        # logger.info("Attention complete")
        flags.pantilt.clear()


# def fetch_events(
#     sink=None,
#     window=None,
#     drop_rate=None,
#     events_lock=None,
#     numevs=None,
#     flags: Flags = None,
# ):
#     logger.info("Fetch_events started.")
#     while not flags.halt.is_set():
#         logger.info("Waiting for attention to finish...")
#         flags.events.wait()

#         if flags.halt.is_set():
#             break
#         logger.info("Accumulating events...")
#         # time.sleep(random.gauss(1, 0.05))
        
#         events = sink.get_events_blocking(1000)  # ms
#         if events:
#             filtered_events = [event for event in events if random.random() > drop_rate]
#             with events_lock:
#                 if filtered_events:
#                     window[
#                         [event.y for event in filtered_events],
#                         [event.x for event in filtered_events],
#                     ] = 255
#                     numevs[0] += len(filtered_events)
        
#             flags.events.clear()
#             flags.attention.set()

#         else:
#             logger.info("No events, exiting...")
#             flags.halt.set()

def clean_up():    
    flags.halt.set()
    flags.events.set()
    flags.pantilt.set()
    

if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    flags = Flags()

    ##### egomotion #####

    # loading egomotion kernel
    filter_egomotion = egokernel()

    # Initialize the network with the loaded filter
    netegomotion = net_def(filter_egomotion, tau_mem, num_pyr, filter_egomotion.size(1))

    ##### attention #####

    # loading attention kernels
    filters_attention = create_vm_filters(thetas, size, rho, r0, thick, offset)
    # plot_filters(filters_attention, thetas)

    # Initialize the attention network with the loaded filters
    netattention = network_init(filters_attention)

    #### PanTilt setup ####

    # Define the serial port and settings
    serial_port = "/dev/tty.usbserial-FTGZO55F"
    baud_rate = 9600

    # Define pan and tilt range values
    pan_range = np.array([
        -3090,
        3090,
    ])  # Replace with actual pan range of your device if different
    tilt_range = np.array([
        -907,
        604,
    ])  # Replace with actual tilt range of your device if different

    ##### Speck initialisation for the sink #####
    # Set the Speck and sink events
    sink, window, numevs, events_lock = Specksetup()
    ####### if we do not have the speck
    # sink = None
    # window = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
    # numevs = [0]  # Use a list to allow modification within the thread
    # events_lock = threading.Lock()

    # A switch for emulating pan/tilt motion
    showstats = False
    pantiltunit = True
    dummy_pan_tilt = False

    # Start the event-fetching thread
    # event_thread = threading.Thread(
    #     target=fetch_events, args=(sink, window, drop_rate, events_lock, numevs, flags)
    # )
    # event_thread.daemon = True
    # event_thread.start()

    suppression = torch.zeros((1, max_y, max_x), device=device)

    attention_thread = threading.Thread(
        target=perform_attention_with_pan_tilt,
        args=(dummy_pan_tilt, showstats, flags, suppression, pan_range, tilt_range),
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
            
                # logger.info("Waiting for attention to finish...")

                if flags.halt.is_set():
                    break
                # logger.info("Accumulating events...")
                # time.sleep(random.gauss(1, 0.05))
                
                events = sink.get_events_blocking(2000)  # ms
                if events:
                    filtered_events = [event for event in events if random.random() > drop_rate]
                    with events_lock:
                        if filtered_events:
                            window[:] = 0
                            window[
                                [event.y for event in filtered_events],
                                [event.x for event in filtered_events],
                            ] = 255
                            window[:] = np.flipud(window)
                            numevs[0] += len(filtered_events)

                            # logger.info("Attention started")


                    # logger.info("Computing egomotion")
                    # time.sleep(random.gauss(1, 0.05))
                    indexes = egomotion(window, netegomotion, numevs, device, suppression)

                    # logger.info("Computing attention")
                    salmap, salmap_coords[:] = run_attention(suppression.detach().cpu().numpy(), netattention, device)

                    flags.pantilt.set()

                    cv2.imshow('Events', window)
                    cv2.imshow('Events after Suppression', suppression[0].detach().cpu().numpy())
                    cv2.circle(salmap, (salmap_coords[1], salmap_coords[0]), 5, (255, 255, 255), -1)
                    cv2.imshow('Saliency Map', cv2.applyColorMap(cv2.convertScaleAbs(salmap), cv2.COLORMAP_JET))
                    cv2.waitKey(1)
                
                else:
                    logger.info("No events, exiting...")
                    flags.halt.set()

        except KeyboardInterrupt:
            flags.halt.set()
            break

        finally:
            clean_up()
            event_thread.join()
            attention_thread.join()
