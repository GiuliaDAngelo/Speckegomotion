'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script visualize the events from the DVS sensor.
'''


import numpy as np
import cv2
import sinabs.backend.dynapcnn.io as sio
import samna
import time
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

