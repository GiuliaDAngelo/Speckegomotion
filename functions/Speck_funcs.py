'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz
Functions for fetching events from the Speck2f camera and setting up the camera for event streaming.
'''
import numpy as np
import sinabs.backend.dynapcnn.io as sio
import samna
import threading
import random


def fetch_events(sink, window, drop_rate, events_lock, numevs):
    while True:
        events = sink.get_events_blocking(1000)
        if events:
            filtered_events = [event for event in events if random.random() > drop_rate]
            with events_lock:
                if filtered_events:
                    window[[event.y for event in filtered_events], [event.x for event in filtered_events]] = 255
                    numevs[0] += len(filtered_events)

def Specksetup(resolution):
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
    numevs = [0]  # Use a list to allow modification within the thread
    events_lock = threading.Lock()
    return sink, window, numevs, events_lock


