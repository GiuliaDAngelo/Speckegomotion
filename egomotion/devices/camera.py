# --------------------------------------
import sinabs.backend.dynapcnn.io as sio

# --------------------------------------
import numpy as np

# --------------------------------------
import samna

# --------------------------------------
import random

# --------------------------------------
from egomotion.conf import logger
from egomotion import conf
from egomotion.utils import Flags

# NOTE: Samna seems to support Davis 346, do we need dv_processing?
# import dv_processing as dv
# # Open the specified camera
# capture = dv.io.CameraCapture(cameraName="DAVIS 346")
# print("end")


class SpeckDevice:
    def __init__(
        self,
        verbose: bool = True,
        drop_rate: float = 0.0,
    ):

        # Extra logging
        self.verbose = verbose

        # Event drop rate
        self.drop_rate = drop_rate

        # Create an empty frame for event visualization
        self.frame = np.zeros((conf.vis_resolution[1], conf.vis_resolution[0]), dtype=np.uint8)
        self.n_events = 0  # Use a list to allow modification within the thread

        # Speck configuration
        # ==================================================
        # List all connected devices
        device_map = sio.get_device_map()

        if self.verbose:
            logger.debug(device_map)

        # Open the devkit device
        devkit = sio.open_device("speck2fdevkit:0")

        # Create and configure the event streaming graph
        samna_graph = samna.graph.EventFilterGraph()
        devkit_config = samna.speck2f.configuration.SpeckConfiguration()
        devkit_config.dvs_layer.raw_monitor_enable = True
        devkit.get_model().apply_configuration(devkit_config)

        self.sink = samna.graph.sink_from(devkit.get_model_source_node())
        samna_graph.start()
        devkit.get_stop_watch().start()
        devkit.get_stop_watch().reset()

    def get_events(
        self,
        window: int = 1000,
        filter: bool = True,
        store: bool = True,
    ):
        events = self.sink.get_events_blocking(window)  # us

        if filter:
            # REVIEW: This is not very efficient.
            # It can probably be replaced with something like np.choice(...).
            filtered_events = [
                event for event in events if random.random() > conf.vis_drop_rate
            ]

        if store:
            self.frame[:] = 0
            self.frame[
                [event.y for event in filtered_events],
                [event.x for event in filtered_events],
            ] = 255
            self.frame[:] = np.flipud(self.frame)
            self.n_events += len(filtered_events)

        return events

    def fetch_events(
        self,
        window=None,
        drop_rate=None,
        events_lock=None,
        flags: Flags = None,
    ):
        logger.info("Fetch_events started.")
        while not flags.halt.is_set():
            logger.info("Waiting for attention to finish...")
            flags.events.wait()

            if flags.halt.is_set():
                break
            logger.info("Accumulating events...")
            # time.sleep(random.gauss(1, 0.05))

            events = self.sink.get_events_blocking(window)  # ms
            if events:
                filtered_events = [
                    event for event in events if random.random() > drop_rate
                ]
                with events_lock:
                    if filtered_events:
                        window[
                            [event.y for event in filtered_events],
                            [event.x for event in filtered_events],
                        ] = 255
                        self.n_events += len(filtered_events)

                flags.events.clear()
                flags.attention.set()

            else:
                logger.info("No events, exiting...")
                flags.halt.set()
