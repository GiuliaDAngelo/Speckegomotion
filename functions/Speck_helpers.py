'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script visualize the events from the DVS sensor.
'''
import numpy
import numpy as np
from scipy.special import iv
import cv2
import sinabs.backend.dynapcnn.io as sio
import samna
import time
import threading
import random
from configSpeckmain import *


def send_command(ser, command):
    ser.write(command.encode('utf-8'))
    time.sleep(0.1)  # Small delay to allow the device to process the command



def fetch_events(sink, window, drop_rate, events_lock, numevs):
    while True:
        events = sink.get_events_blocking(1000) #ms
        if events:
            filtered_events = [event for event in events if random.random() > drop_rate]
            with events_lock:
                if filtered_events:
                    window[[event.y for event in filtered_events], [event.x for event in filtered_events]] = 255
                    numevs[0] += len(filtered_events)

def Specksetup():
    # List all connected devices
    device_map = sio.get_device_map()
    print(device_map)

    # Open the devkit device
    devkit = sio.open_device("speck2fdevkit:0")

    # Create and configure the event streaming graph
    samna_graph = samna.graph.EventFilterGraph()
    devkit_config = samna.speck2f.configuration.SpeckConfiguration()
    devkit_config.dvs_layer.raw_monitor_enable = True
    devkit.get_model().apply_configuration(devkit_config)
    sink = samna.graph.sink_from(devkit.get_model_source_node())
    samna_graph.start()
    devkit.get_stop_watch().start()
    devkit.get_stop_watch().reset()

    # Create an empty window for event visualization
    window = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
    window_pos = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
    window_neg = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
    numevs = [0]  # Use a list to allow modification within the thread
    events_lock = threading.Lock()
    return sink, window, window_pos, window_neg, numevs, events_lock

