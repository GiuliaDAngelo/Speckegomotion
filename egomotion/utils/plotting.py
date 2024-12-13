# --------------------------------------
import os

# --------------------------------------
import cv2

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from pathlib import Path

# --------------------------------------
import numpy as np

# --------------------------------------
import torch

# --------------------------------------
from natsort import natsorted


def create_video_from_frames(
    frame_dir: Path | str,
    output_video: Path | str,
    fps: int = 30,
):
    """
    Create a video from a series of frames.

    Args:
        frame_dir (Path | str):
            The directory where the frames are stored.

        file_path (Path | str):
            The path to the output video file.

        fps (int, optional):
            Frames per second. Defaults to 30.
    """
    # Get all the frame file names and sort them
    frames = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    # frames.sort()  # Ensure the frames are in order
    frames = natsorted(frames)

    # Get the width and height of the first frame
    first_frame_path = os.path.join(frame_dir, frames[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_name in frames:
        frame_path = os.path.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path)
        video.write(frame)  # Write the frame to the video

    video.release()  # Close the video writer
    cv2.destroyAllWindows()


def plot_runningmean(
    numframes: int,
    time_wnd_frames: np.ndarray,
    meanegomap: np.ndarray,
    respath: Path | str,
    title: str = "",
):
    """
    Plot the running mean of a sequence of frames.

    Args:
        numframes (int):
            The numnber of frames in the sequence.

        time_wnd_frames (np.ndarray):
            REVIEW: Is this needed?
            Time window frames.

        meanegomap (np.ndarray):
            The mean of the egomap.

        respath (Path | str):
            Path to the directory where the results should be saved.

        title (str, optional):
            Figure title.
                Defaults to "".
    """
    print(meanegomap)
    time = np.arange(
        0, (numframes - 1) * time_wnd_frames * 1e-3, time_wnd_frames * 1e-3
    )
    plt.plot(time, meanegomap)
    plt.xlabel("Time [ms]")  # plotting by columns
    plt.ylabel("Running mean - network activity")
    # plt.show()
    plt.savefig(respath + "/" + title, dpi=300)  # Save as PNG with 300 dpi


def plot_kernel(
    kernel: np.ndarray,
):
    """
    Plot a Gaussian kernel in 3D.

    Args:
        kernel (np.ndarray):
            The kernel to plot.
    """
    # plot kernel 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = torch.linspace(-kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0])
    y = torch.linspace(-kernel.shape[1] // 2, kernel.shape[1] // 2, kernel.shape[1])
    x, y = torch.meshgrid(x, y, indexing="ij")
    ax.plot_surface(x.numpy(), y.numpy(), kernel.numpy(), cmap="jet")
    plt.show()


def plot_filters(
    filters: np.ndarray,
    angles: list[float],
):
    """
    Plot the von Mises filters.

    Args:
        filters (torch.Tensor):
            A tensor containing filters to be visualized.

        angles (torch.Tensor):
            Angles of the corresponding the filters.
    """
    # Create subplots for 8 orientation VM filters
    # REVIEW: Make this a bit more generic.
    # The number of filters is assumed and hard-coded.
    fig, axes = plt.subplots(2, filters.shape[0] // 2, figsize=(10, 5))
    fig.suptitle(
        f"VM filters size ({filters.shape[1]},{filters.shape[2]})", fontsize=16
    )

    # Display filters with their corresponding angles
    for i, flt in enumerate(filters):
        if i < filters.shape[0] // 2:
            axes[0, i].set_title(f"{round(angles[i],2)} grad")
            axes[0, i].imshow(flt[i])
            plt.colorbar(axes[0, i].imshow(flt[i]))
        else:
            axes[1, i - 4].set_title(f"{round(angles[i],2)} grad")
            axes[1, i - 4].imshow(flt[i])
            plt.colorbar(axes[1, i - 4].imshow(flt[i]))
    # add color bar to see the values of the filters
    plt.show()
