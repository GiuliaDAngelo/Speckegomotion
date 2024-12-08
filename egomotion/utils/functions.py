# --------------------------------------
import os

# --------------------------------------
from pathlib import Path

# --------------------------------------
from datetime import datetime

# --------------------------------------
import numpy as np

# --------------------------------------
import torch


def cwd(path: Path | str) -> Path:
    """
    Return the current working directory of the supplied path.
    Works with both directories and files.

    Args:
        path (Path | str):
            A path being queried.

    Returns:
        Path:
            The directory of that path.
    """
    path = Path(path).expanduser().resolve().absolute()
    if not path.exists():
        return None
    return path.parent if path.is_file() else path


def timestamp(
    ms: bool = False,
    fmt: str = "%Y-%m-%d_%H-%M-%S",
) -> str:
    """
    Create a timestamp as a string.

    Args:
        ms (bool, optional):
            Indicates whether the timestamp should include milliseconds.
            Defaults to False.

        fmt: (str, optional):
            The format to use.
            Defaults to "%Y-%m-%d_%H-%M-%S".

    Returns:
        str:
            The timestamp in the specified format.
    """

    end = None

    if ms:
        # With ms precision
        fmt += ":%f"
        end = -3

    return datetime.strftime(datetime.now(datetime.UTC), fmt)[:end]


def mkdir(
    name: str | Path,
) -> Path:
    """
    Create a new directory and return the Path object.

    Args:
        name (str | Path):
            The name of the new directory.

    Returns:
        Path:
            The new path.
    """
    name = Path(name).expanduser().resolve().absolute()
    name.mkdir(exist_ok=True, parents=True)
    return name


def atan2(
    x: float,
    y: float,
):
    """
    Compute the angle in radians between the positive x-axis and the point (x, y),
    ensuring the angle is in the range [0, 2π].

    Args:
        x (float):
            x-coordinate of the point.

        y (float):
            y-coordinate of the point.

    Returns:
        angle (float):
            Angle in radians, between 0 and 2π.
    """
    # Get the angle in radians and wrap it in the range [0, 2π]
    return np.arctan2(y, x) % (2 * np.pi)


def ems(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    alpha: float = 0.1,
    eps: float = 1e-9,
) -> tuple[torch.Tensor]:
    """
    Exponential moving statistics.

    Args:
        x (torch.Tensor):
            The new input.

        mean (torch.Tensor):
            The current mean.

        var (torch.Tensor):
            The current variance.

        alpha (float, optional):
            The forgetting factor. Defaults to 0.1.

        eps (float, optional):
            A small constant to avoid division by 0. Defaults to 1e-9.

    Returns:
        tuple[torch.Tensor]:
            A tuple containing:
                - The new mean.
                - The new variance.
                - The new SD.
    """

    diff = x - mean
    inc = alpha * diff
    mean += inc
    var = (1.0 - alpha) * (var + diff * inc)
    sd = torch.sqrt(var + eps)

    return (mean, var, sd)


def pyr_res(num_pyr, frames):
    '''
    REVIEW: Unclear what this function does.

    Args:
        num_pyr (_type_):
            _description_

        frames (_type_):
            _description_

    Returns:
        _type_:
            _description_
    '''
    # REVIEW: What does this function do?
    res = []
    for pyr in range(1, num_pyr + 1):
        res.append(
            (int((frames[0][0].shape[0]) / pyr), int((frames[0][0].shape[1]) / pyr))
        )
    return res
