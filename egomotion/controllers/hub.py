# --------------------------------------
import matplotlib

matplotlib.use("TkAgg")

# --------------------------------------
import torch

# --------------------------------------
import torchvision

# --------------------------------------
import cv2 as cv

# --------------------------------------
import numpy as np

# --------------------------------------
import time

# --------------------------------------
import skimage as ski

# --------------------------------------
from datetime import datetime

# --------------------------------------
import threading

# --------------------------------------
from egomotion import conf
from egomotion.conf import logger
from egomotion.network import AttentionNet
from egomotion.controllers import NengoController
from egomotion.devices import PTU
from egomotion.devices import SpeckDevice
from egomotion.utils import Flags


class Hub:

    def __init__(
        self,
        net: AttentionNet,
        ptu: PTU,
        speck: SpeckDevice,
        controller: NengoController,
        alpha: float = 0.9,
        stats: bool = True,
    ):

        self.flags = Flags()

        # REVIEW: What is this?
        self.alpha = alpha

        # Event frame
        self.ev_frame = np.zeros(conf.sensor_size, dtype=np.ubyte)

        # Suppression map
        self.suppression_map = np.zeros_like(self.ev_frame)

        # Saliency map coordinates
        self.saliency_map = np.zeros_like(self.ev_frame)
        self.saliency_map_coords = np.array([0, 0])

        # Egomotion and attention
        # ==================================================
        self.net = net

        # Pan/tilt unit (PTU)
        # ==================================================
        self.ptu = ptu
        self.ptu.reset()
        self.ptu_thread = threading.Thread(target=self.follow_with_attention)
        self.ptu_thread.daemon = True

        # Nengo controller for the PTU
        # REVIEW: Check if the controller is instantiated
        # with the right arguments.
        # ==================================================
        self.controller = controller

        # Speck camera
        # ==================================================
        self.speck = speck

        if stats:
            num_eve = []
            num_eve_supp = []
            num_eve_drop = []
            attention_maps = []
            salmap_coordinates = []
            windows = []

    def follow_with_attention(self):

        pan_norm = (self.ptu.pan_range[1] - self.ptu.pan_range[0]) / 2
        tilt_norm = (self.ptu.tilt_range[1] - self.ptu.tilt_range[0]) / 2

        while not self.flags.halt.is_set():

            logger.debug("PTU waiting for attention trigger...")

            self.flags.attention.wait()

            if self.flags.halt.is_set():
                break

            # REVIEW: The controller signature doesn't really match this call
            # cmds values are between -1 and 1
            # alpha = 0.9
            # cmd = run_controller(
            #     np.array([self.salmap_coords[0], self.salmap_coords[1]]),
            #     np.array([self.resolution[1] // 2, self.resolution[0] // 2]),
            #     k_pan=np.array([self.alpha, 0.0, 0.0]),
            #     k_tilt=np.array([self.alpha, 0.0, 0.0]),
            # )
            cmd = self.controller(self.saliency_map_coords)

            delta_pan = int(2 * cmd[0] * pan_norm / conf.sensor_size[0])
            delta_tilt = -int(2 * cmd[1] * tilt_norm / conf.sensor_size[1])

            # make a check if pan_angle and tilt_angle are within the range
            pan_angle = np.clip(
                delta_pan,
                self.ptu.pan_range[0],
                self.ptu.pan_range[1],
            ).astype(np.int32)

            tilt_angle = np.clip(
                delta_tilt,
                self.ptu.tilt_range[0],
                self.ptu.tilt_range[1],
            ).astype(np.int32)

            # Dummy pan & tilt
            if conf.DUMMY:
                time.sleep(np.random.uniform(0.1, 0.5))
            else:
                response = self.ptu.move(pan_angle, tilt_angle)
                logger.debug(f"Hub | Response: {response}")

            logger.debug("Hub | Movement complete")
            self.flags.attention.clear()

    def run(self):

        self.ptu_thread.start()
        self.flags.attention.set()
        while not self.flags.halt.is_set():

            current_time = datetime.now()
            logger.info("Hub | Running...")

            try:
                while True:

                    # logger.info("Waiting for attention to finish...")

                    if self.flags.halt.is_set():
                        break
                    # logger.info("Accumulating events...")
                    # time.sleep(random.gauss(1, 0.05))

                    (events, self.frame) = self.speck.get_events(1000)  # us

                    if events:

                        # logger.info("Computing egomotion")
                        # time.sleep(random.gauss(1, 0.05))
                        self.compute_egomotion()

                        # logger.info("Computing attention")
                        self.compute_attention()

                        # Visualise the events and the saliency map.
                        self.show_events()

                        self.flags.attention.set()

                    else:
                        logger.info("Hub | No more events, exiting...")
                        self.flags.halt.set()

            except KeyboardInterrupt:
                self.flags.halt.set()
                break

            finally:
                self.clean_up()
                self.ptu_thread.join()

    def show_events(self):
        """
        Visualise the event frame, the suppression map and the saliency map.
        """

        cv.imshow("Events", self.ev_frame)
        cv.imshow(
            "Events after suppression",
            self.suppression_map[0],
        )
        cv.circle(
            self.saliency_map,
            (self.saliency_map_coords[1], self.saliency_map_coords[0]),
            5,
            (255, 255, 255),
            -1,
        )
        cv.imshow(
            "Saliency Map",
            cv.applyColorMap(cv.convertScaleAbs(self.saliency_map), cv.COLORMAP_JET),
        )
        cv.waitKey(1)

    def compute_egomotion(self):
        """
        Extract an egomotion suppression map from an event frame.
        """

        # Turn the window into a tensor
        torch_frame = torch.from_numpy(self.frame).unsqueeze(0).float().to(conf.device)

        # Compute the egomap
        with torch.no_grad():
            egomap = self.net.compute_egomotion(torch_frame)

        # Resize the egomap to match the resolution of the input
        # REVIEW This might be the source of the slightly
        # lower performance in terms of IoU
        # ==================================================
        # egomap = torch.nn.functional.interpolate(
        #     egomap.unsqueeze(0),
        #     size=(conf.vis_max_y, conf.vis_max_x),
        #     mode="bilinear",
        #     align_corners=False,
        # ).squeeze()

        # Convert the egomap to a NumPy array
        egomap = egomap.squeeze().detach().numpy()

        # frame, egomap between 0 and 255
        # egomap = 255 * (egomap - egomap.min()) / (egomap.max() - egomap.min())
        # frame = 255 * (window - window.min()) / (window.max() - window.min())
        # REVIEW This might be the wrong thing to do here because
        # it erases differences in relative dynamic range across frames.
        # REVIEW: Try without any scaling
        egomap = ski.exposure.rescale_intensity(
            egomap, in_range="image", out_range=np.uint8
        )
        frame = ski.exposure.rescale_intensity(
            self.frame, in_range="image", out_range=np.uint8
        )

        # suppression = torch.zeros((1, conf.max_y, conf.max_x), device=conf.device)

        # Find where the egomap is over the threashold suppression max = frame
        indices = egomap >= conf.threshold

        # REVIEW: There is no need for the extra frame variable at all
        # suppression[indexes] = frame[indexes]
        self.suppression_map = np.zeros_like(egomap)
        self.suppression_map[indices] = 1

    def compute_attention(self):
        """
        Perform attention over a suppression map as computed
        by the egomotion process.
        """

        # Create resized versions of the frames

        resized_frames = [
            torchvision.transforms.Resize(
                # (int(suppression_map.shape[2] / pyr), int(suppression_map.shape[1] / pyr))
                (
                    # REVIEW: The original gave the wrong scaling
                    # It should be a power of 2 (so 2 ** pyr), but it was just `pyr`.
                    self.suppression_map.shape[0] // (2**pyr),
                    self.suppression_map.shape[1] // (2**pyr),
                )
            )(torch.from_numpy(self.suppression_map).unsqueeze(0))
            for pyr in range(conf.vm_num_pyr)
        ]

        # Process frames in batches
        # REVIEW: Should be vstack as stack gives an unsqueezed tensor.
        batch_frames = torch.vstack(
            [
                torchvision.transforms.Resize(
                    (conf.sensor_size[0], conf.sensor_size[1])
                )(frame)
                for frame in resized_frames
            ]
        ).type(torch.float32)

        with torch.no_grad():
            batch_frames = batch_frames.to(conf.device)  # Move to GPU if available
            attention_response = self.net.compute_attention(batch_frames)

        # Sum the outputs over rotations and scales
        saliency_map = (
            (
                torch.sum(
                    torch.sum(attention_response, dim=1, keepdim=True),
                    dim=0,
                    keepdim=True,
                )
                .squeeze()
                .type(torch.float32)
            )
            .detach()
            .numpy()
        )
        saliency_map_coords = np.unravel_index(
            np.argmax(saliency_map), saliency_map.shape
        )

        # normalise salmap for visualization
        # saliency_map = np.array(
        #     (saliency_map - saliency_map.min())
        #     / (saliency_map.max() - saliency_map.min())
        #     * 255
        # )
        saliency_map = ski.exposure.rescale_intensity(
            saliency_map, in_range="image", out_range=np.uint8
        )

        # rescale salmap to the original size
        # saliency_map = resize(saliency_map, (self.frame.shape), anti_aliasing=False)

        self.saliency_map = saliency_map
        self.saliency_map_coords = saliency_map_coords

    def clean_up(self):
        self.flags.halt.set()
        self.flags.attention.set()

    # def process(filter, frames, max_x, max_y, time_wnd_frames):

    #     # Define motion parameters
    #     tau_mem = time_wnd_frames * 10**-3  # tau_mem in milliseconds
    #     # Initialize the network with the loaded filter
    #     net = net_def(filter, tau_mem, num_pyr)
    #     cnt = 0
    #     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #     for frame in frames:
    #         print(str(cnt) + " frame out of " + str(frames.shape[0]))
    #         frame = frame.to(conf.device, dtype=net[0].weight.dtype)
    #         egomap = net(frame)
    #         # resize egomap to the original size
    #         egomap = torch.nn.functional.interpolate(
    #             egomap.unsqueeze(0),
    #             size=(max_y + 1, max_x + 1),
    #             mode="bilinear",
    #             align_corners=False,
    #         ).squeeze(0)
    #         # frame, egomap between 0 and 255
    #         frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
    #         egomap = (egomap - egomap.min()) / (egomap.max() - egomap.min()) * 255
    #         # values under a threashold are set to 0
    #         egomap[egomap < threshold] = 0
    #         # create suppression map
    #         suppression = torch.zeros((1, max_y + 1, max_x + 1), device=device)
    #         # where egomap is over the threashold suppression max = frame
    #         suppression[egomap >= threshold] = frame[egomap >= threshold]

    #         # Show the egomap
    #         if show_egomap:
    #             # plot the frame and overlap the max point of the saliency map with a red dot
    #             plt.clf()
    #             # subplot ahowing the frame and the egomap
    #             plt.subplot(1, 3, 1)
    #             plt.imshow(frame.squeeze(0).cpu().detach().numpy(), cmap="gray")
    #             plt.colorbar(shrink=0.3)
    #             plt.title("Frame")

    #             plt.subplot(1, 3, 2)
    #             plt.imshow(egomap.squeeze(0).cpu().detach().numpy(), cmap="jet")
    #             plt.colorbar(shrink=0.3)
    #             plt.title("Egomap Map")

    #             # plot suppression map
    #             plt.subplot(1, 3, 3)
    #             plt.imshow(suppression.squeeze(0).cpu().detach().numpy(), cmap="gray")
    #             plt.colorbar(shrink=0.3)
    #             plt.title("Suppression Map")

    #             plt.draw()
    #             plt.pause(0.001)
    #         if save_res:
    #             # save the plot in a video
    #             plt.savefig(respath + "egomaps/egomap" + str(cnt) + ".png")
    #         cnt += 1
