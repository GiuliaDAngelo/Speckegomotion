# --------------------------------------
import torch
from torch import nn


class WinnerTakesAll(nn.Module):
    def __init__(
        self,
        k: int = 1,  # What is this?
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.k = k

    def forward(self, x: torch.Tensor):
        # Flatten the input except for the batch dimension
        flat_x = x.view(x.size(0), -1)
        # Get the top-k values and their indices
        topk_vals, topk_indices = torch.topk(flat_x, self.k, dim=1)
        # Create a mask of the same shape as flat_x
        mask = torch.zeros_like(flat_x)
        # Set the top-k values in the mask to 1
        mask.scatter_(1, topk_indices, 1)
        # Reshape the mask to the original input shape
        mask = mask.view_as(x)
        # Apply the mask to the input
        return x * mask
