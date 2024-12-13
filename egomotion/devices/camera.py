# --------------------------------------
import numpy as np

# --------------------------------------
import samna

# --------------------------------------
import sinabs.backend.dynapcnn.io as sio

# --------------------------------------
import time

# --------------------------------------
from tonic.transforms import DropEvent
from tonic.transforms import ToFrame

# --------------------------------------
from egomotion.conf import logger
from egomotion import conf

# NOTE: Samna seems to support Davis 346, do we need dv_processing?
# import dv_processing as dv
# # Open the specified camera
# capture = dv.io.CameraCapture(cameraName="DAVIS 346")
# print("end")


class SpeckDevice:
    def __init__(
        self,
        window: int = 1000,
    ):
        # The event accumulation window
        self.window = window

        # The number of accumulated events
        self.n_events = 0

        self.ev_drop_transform = DropEvent(p=conf.ev_drop_rate)
        self.frame_transform = ToFrame(
            sensor_size=conf.sensor_size,
            time_window=self.window,
        )

        # Speck configuration
        # ==================================================
        if not conf.DUMMY:
            # List all connected devices
            device_map = sio.get_device_map()

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
        filter: bool = True,
    ):

        logger.info("Accumulating events...")

        if conf.DUMMY:
            time.sleep(np.random.uniform(self.window / 1000))

            events = True
            frame = np.random.binomial(1, conf.ev_random_rate, size=conf.sensor_size)

        else:
            events = self.sink.get_events_blocking(self.window)  # us

            if filter:
                # REVIEW: This is not very efficient.
                # It can probably be dropped with Tonic.
                # events = [
                #     event for event in events if random.random() > conf.vis_drop_rate
                # ]
                events = self.ev_drop_transform(events)

            idx = np.array([[event.x, event.y] for event in events])
            frame = np.zeros(conf.sensor_size)
            frame[idx[:, 0], idx[:, 1]] = 255
            # self.frame[:] = np.flipud(self.frame)
            self.n_events += len(events)

        return (events, frame)

    # def fetch_events(
    #     self,
    #     window: int = None,
    #     drop_rate: float = None,
    #     flags: Flags = None,
    # ):

    #     while not flags.halt.is_set():
    #         logger.info("Waiting for attention to finish...")
    #         flags.events.wait()

    #         if flags.halt.is_set():
    #             break

    #         events = self.sink.get_events_blocking(window)  # us
    #         if events:
    #             filtered_events = [
    #                 event for event in events if random.random() > drop_rate
    #             ]

    #             if filtered_events:
    #                 window[
    #                     [event.y for event in filtered_events],
    #                     [event.x for event in filtered_events],
    #                 ] = 255
    #                 self.n_events += len(filtered_events)

    #             flags.events.clear()
    #             flags.attention.set()

    #             yield filtered_events, window

    #         else:
    #             logger.info("No events, exiting...")
    #             flags.halt.set()
