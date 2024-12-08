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
        center_kernel: torch.Tensor,
        surround_kernel: torch.Tensor,
        attention_kernel: torch.Tensor,
        alpha: float = 0.9,
        stats: bool = True,
        verbose: bool = True,
    ):

        # Extra logging
        self.verbose = verbose

        self.flags = Flags()

        # REVIEW: What is this?
        self.alpha = alpha

        # Egomotion and attention
        # ==================================================
        self.net = AttentionNet(
            center_kernel,
            surround_kernel,
            attention_kernel,
            conf.tau_mem,
            conf.vm_num_pyr,
        )

        # Nengo controller for the PTU
        # REVIEW: Check if the controller is instantiated
        # with the right arguments.
        # ==================================================
        self.controller = NengoController(
            np.array([self.salmap_coords[0], self.salmap_coords[1]]),
            np.array([conf.vis_resolution[1] // 2, conf.vis_resolution[0] // 2]),
            k_pan=np.array([self.alpha, 0.0, 0.0]),
            k_tilt=np.array([self.alpha, 0.0, 0.0]),
        )

        # Pan/tilt unit (PTU)
        # ==================================================
        self.ptu = PTU()
        self.ptu.reset()
        self.ptu_thread = threading.Thread(target=self.follow_with_attention)
        self.ptu_thread.daemon = True
        self.ptu_thread.start()

        # Speck camera
        # ==================================================
        self.speck = SpeckDevice()

        # Saliency map coordinates
        self.salmap = None
        self.salmap_coords = np.array([0, 0])

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

            if self.verbose:
                logger.info("PTU waiting for attention trigger...")

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
            cmd = self.controller(self.salmap_coords)

            delta_pan = int(2 * cmd[0] * pan_norm / conf.vis_resolution[0])
            delta_tilt = -int(2 * cmd[1] * tilt_norm / conf.vis_resolution[1])

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
            if conf.dummy:
                time.sleep(np.random.uniform(0.1, 0.5))
            else:
                response = self.ptu.move(pan_angle, tilt_angle)
                if self.verbose:
                    logger.debug(f"Hub | Response: {response}")

            if self.verbose:
                logger.info("Hub | Movement complete")
            self.flags.attention.clear()

    def run(self):

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

                    events = self.speck.get_events(1000)  # us

                    if events:

                        # logger.info("Computing egomotion")
                        # time.sleep(random.gauss(1, 0.05))
                        suppression, indices = self.compute_egomotion(self.speck.frame)

                        # logger.info("Computing attention")
                        self.salmap, self.salmap_coords[:] = self.compute_attention(
                            suppression
                        )

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
        Visualise the events.
        """

        cv.imshow("Events", self.speck.frame)
        cv.imshow(
            "Events after Suppression",
            self.suppression[0].detach().cpu().numpy(),
        )
        cv.circle(
            self.salmap,
            (self.salmap_coords[1], self.salmap_coords[0]),
            5,
            (255, 255, 255),
            -1,
        )
        cv.imshow(
            "Saliency Map",
            cv.applyColorMap(cv.convertScaleAbs(self.salmap), cv.COLORMAP_JET),
        )
        cv.waitKey(1)

    def compute_egomotion(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract an egomotion suppression map from an event frame.

        Args:
            frame (np.ndarray):
                The current event frame.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing:
                    - The reduced map.
                    - The indices of pixels where the intensity is over the threshold.
        """

        # Turn the window into a tensor
        frame = torch.from_numpy(frame).unsqueeze(0).float().to(conf.device)

        # Compute the egomap
        egomap = self.net.compute_egomotion(frame)

        # Resize the egomap to match the resolution of the input
        # REVIEW This might be the source of the slightly
        # lower performance in terms of IoU
        # ==================================================
        egomap = torch.nn.functional.interpolate(
            egomap.unsqueeze(0),
            size=(conf.vis_max_y, conf.vis_max_x),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

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
            frame, in_range="image", out_range=np.uint8
        )

        # suppression = torch.zeros((1, conf.max_y, conf.max_x), device=conf.device)

        # Find where the egomap is over the threashold suppression max = frame
        indexes = egomap >= conf.threshold
        # REVIEW: There is no need for the frame variable at all
        # suppression[indexes] = frame[indexes]
        # suppression = np.ones_like(frame)[indexes]
        suppression = np.ones_like(frame)[indexes]

        return suppression, indexes

    def compute_attention(
        self,
        window: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform attention over a reduced map as computed
        by the egomotion process.

        Args:
            window (np.ndarray):
                The reduced window.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing:
                    - The saliency map.
                    - The coordinates of the most salient feature.
        """

        # Create resized versions of the frames
        resized_frames = [
            torchvision.transforms.Resize(
                (int(window.shape[2] / pyr), int(window.shape[1] / pyr))
            )(torch.from_numpy(window))
            for pyr in range(1, conf.vm_num_pyr + 1)
        ]

        # Process frames in batches

        batch_frames = torch.stack(
            [
                torchvision.transforms.Resize((conf.vis_resolution[0], conf.vis_resolution[1]))(
                    frame
                )
                for frame in resized_frames
            ]
        ).type(torch.float32)
        batch_frames = batch_frames.to(conf.device)  # Move to GPU if available
        output_rot = self.net.compute_attention(batch_frames)

        # Sum the outputs over rotations and scales
        salmap = (
            torch.sum(torch.sum(output_rot, dim=1, keepdim=True), dim=0, keepdim=True)
            .squeeze()
            .type(torch.float32)
        )
        salmax_coords = np.unravel_index(
            torch.argmax(salmap).cpu().numpy(), salmap.shape
        )

        # normalise salmap for visualization
        salmap = salmap.detach().cpu()
        salmap = np.array((salmap - salmap.min()) / (salmap.max() - salmap.min()) * 255)

        # rescale salmap to the original size
        # salmap = resize(salmap, (window.shape[1], window.shape[2]), anti_aliasing=False)
        return (salmap, salmax_coords)

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
