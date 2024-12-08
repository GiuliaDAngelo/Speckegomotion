# --------------------------------------
import torch

# --------------------------------------
import numpy as np

# --------------------------------------
from scipy.special import iv

# --------------------------------------
from pathlib import Path

# --------------------------------------
from skimage.transform import rescale

# --------------------------------------
from egomotion import conf
from egomotion import utils


def gaussian(
    size: int,
    sigma: float,
    norm: bool = True,
) -> torch.Tensor:
    """
    Create a grid of (x, y) coordinates.

    Args:
        size (int):
            The size of the kernel.

        sigma (float):
            The spread (standard deviation) of the kernel.

        norm (bool, optional):
            Indicates whether the kernel should be normalised.
            Defaults to True.

    Returns:
        torch.Tensor:
            The kernel.
    """
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y)

    # Create a Gaussian kernel
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

    if norm:
        kernel /= 2 * np.pi * sigma**2

    return kernel.unsqueeze(0).unsqueeze(1)


def von_mises(
    theta: float,
    scale: float,
    rho: float = 0.1,
    r0: float = 0,
    thick: float = 0.5,
    offset: tuple[float, float] = (0, 0),
) -> torch.Tensor:
    """
    Generate a Von Mises filter with r0 shifting and an offset.

    Args:
        theta (_type_):
            _description_

        scale (_type_):
            _description_

        rho (float, optional):
            _description_. Defaults to 0.1.

        r0 (int, optional):
            _description_. Defaults to 0.

        thick (float, optional):
            _description_. Defaults to 0.5.

        offset (tuple, optional):
            _description_. Defaults to (0, 0).

    Returns:
        torch.Tensor:
            The filter.
    """

    height, width = scale, scale
    vm = np.empty((height, width))
    offset_x, offset_y = offset

    for x in range(width):
        for y in range(height):
            # Shift X and Y based on r0 and offset
            X = (x - width / 2) + r0 * np.cos(theta) - offset_x * np.cos(theta)
            Y = (
                (height / 2 - y) + r0 * np.sin(theta) - offset_y * np.sin(theta)
            )  # Inverted Y for correct orientation
            r = np.sqrt(X**2 + Y**2)
            angle = utils.atan2(X, Y)

            # Compute the Von Mises filter value
            vm[y, x] = np.exp(thick * rho * r0 * np.cos(angle - theta)) / iv(0, r - r0)
    # normalise value between -1 and 1
    # vm = vm / np.max(vm)
    # vm = vm * 2 - 1
    return vm


def make_vm_filters(
    thetas: tuple[float],
    size: float,
    rho: float,
    r0: float,
    thick: float,
    offset: tuple[float, float],
) -> torch.Tensor:
    """
    Create a set of Von Mises filters with different orientations.

    Args:
        thetas (np.ndarray):
            Array of angles in radians.

        size (int):
            Size of the filter.

        rho (float):
            Scale coefficient to control arc length.

        r0 (int):
            Radius shift from the center.

    Returns:
        filters (torch.Tensor):
            List of Von Mises filters.
    """
    filters = []
    for theta in thetas:
        filter = von_mises(theta, size, rho=rho, r0=r0, thick=thick, offset=offset)
        filter = rescale(filter, conf.vm_fltr_resize_perc, anti_aliasing=False)
        filters.append(filter)
    filters = torch.tensor(np.stack(filters).astype(np.float32))
    return filters


def load_kernels(path: Path | str) -> torch.Tensor:
    """
    Load saved kernels.

    Args:
        path (Path | str):
            A path to a saved kernel.

    Returns:
        torch.Tensor:
            The loaded kernel.
    """
    filters = []
    filter = np.load(path)
    filter = torch.tensor(np.stack(filter).astype(np.float32))
    filter = filter.unsqueeze(0)
    return filter
