# --------------------------------------
import torch
from torch import nn

# --------------------------------------
import matplotlib

matplotlib.use("TkAgg")
# --------------------------------------
import sinabs.layers as sl

# --------------------------------------
from egomotion import conf


class AttentionNet:

    def __init__(
        self,
        center_filter: torch.Tensor,
        surround_filter: torch.Tensor,
        attention_filter: torch.Tensor,
        tau_mem: float,
        num_pyr: int,
    ):
        # Define the layers and load the filters

        # REVIEW: If an appropriate amount of padding is added
        # to these kernels, the response will have the same
        # shape as the input and the latter won't have to be
        # rescaled downstream.
        # ==================================================
        self.center = nn.Conv2d(
            in_channels=1,
            out_channels=num_pyr,
            kernel_size=(
                conf.k_center_size,
                conf.k_center_size,
            ),
            stride=1,
            padding="same",
            bias=False,
        )
        self.surround = nn.Conv2d(
            in_channels=1,
            out_channels=num_pyr,
            kernel_size=(
                conf.k_surround_size,
                conf.k_surround_size,
            ),
            stride=1,
            padding="same",
            bias=False,
        )

        self.center.weight.data = center_filter.data
        self.surround.weight.data = surround_filter.data

        self.egomotion_lif = sl.LIF(tau_mem)
        # REVIEW: Is this multiplication necesary?
        self.egomotion_lif.v_mem *= self.egomotion_lif.tau_mem

        # Attention
        # ==================================================
        self.attention = nn.Conv2d(
            in_channels=num_pyr,
            # REVIEW: The original didn't have output channels
            out_channels=1,
            kernel_size=(attention_filter.shape[1], attention_filter.shape[2]),
            stride=1,
            padding="same",
            bias=False,
        )

        self.attention.weight.data = attention_filter.data

        self.attention_lif = sl.LIF(tau_mem)
        # REVIEW: Is this multiplication necesary?
        self.attention_lif.v_mem *= self.attention_lif.tau_mem

        # Move all layers to the device
        self.center.to(conf.device)
        self.surround.to(conf.device)
        self.attention.to(conf.device)
        self.egomotion_lif.to(conf.device)
        self.attention_lif.to(conf.device)

    def compute_egomotion(
        self,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the egomotion mechanism.

        Args:
            frame (torch.Tensor):
                The input frame.

        Returns:
            torch.Tensor:
                The egomotion map.
        """

        # Compute the center and surround responses
        center = self.center(frame)
        surround = self.surround(frame)

        # DoG response
        dog = center - surround

        # Run the DoG response through the LIF cells
        response = self.egomotion_lif(dog)

        return response

    def compute_attention(
        self,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the attention mechanism.

        Args:
            frame (torch.Tensor):
                The input frame.

        Returns:
            torch.Tensor:
                The attention response.
        """

        # Compute the attention over the frame
        attention = self.attention(frame)

        # Run the attention response through the LIF cells
        response = self.attention_lif(attention)

        return response
