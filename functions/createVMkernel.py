import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')


def zero_2pi_tan(x, y):
    """
    Compute the angle in radians between the positive x-axis and the point (x, y),
    ensuring the angle is in the range [0, 2π].

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        angle (float): Angle in radians, between 0 and 2π.
    """
    angle = np.arctan2(y, x) % (2 * np.pi)  # Get the angle in radians and wrap it in the range [0, 2π]
    return angle

def vm_filter(theta, scale, rho, r0, offset):
    # theta, scale, rho = 0.1, r0 = 0, thick = 0.5, offset = (0, 0)
    """Generate a Von Mises filter with r0 shifting and an offset."""
    height, width = scale, scale
    vm = np.empty((height, width))
    offset_x, offset_y = offset
    for x in range(width):
        for y in range(height):
            # Shift X and Y based on r0 and offset
            X = (x - width / 2) + r0 * np.cos(theta) + offset_x * np.cos(theta)
            Y = (height / 2 - y) + r0 * np.sin(theta) + offset_y * np.sin(theta)  # Inverted Y for correct orientation
            r = np.sqrt(X**2 + Y**2 - r0)
            angle = zero_2pi_tan(-Y, X)

            # Compute the Von Mises filter value
            # vm[y, x] = np.exp(thick*rho * r0 * np.cos(angle - theta)) / iv(0, r - r0)
            vm[y, x] = np.exp(rho * r0 * np.cos(angle - theta)) / iv(0, r)
    return vm

def create_vm_filters(thetas, size, rho, r0, offset):
    """
    Create a set of Von Mises filters with different orientations.

    Args:
        thetas (np.ndarray): Array of angles in radians.
        size (int): Size of the filter.
        rho (float): Scale coefficient to control arc length.
        r0 (int): Radius shift from the center.

    Returns:
        filters (list): List of Von Mises filters.
    """
    filters = []
    for theta in thetas:
        filter = vm_filter(theta, size, rho=rho, r0=r0, offset=offset)
        filters.append(filter)
    filters = torch.tensor(np.stack(filters).astype(np.float32))
    return filters

def plot_VMkernels(VMkernels,thetas):
    num_filters = len(VMkernels)
    num_rows = (num_filters + (len(VMkernels)//2) - 1) // (len(VMkernels)//2)  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, (len(VMkernels)//2), figsize=(15, 3 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (filter, theta) in enumerate(zip(VMkernels, thetas)):
        ax = axes[i]
        ax.imshow(filter.cpu().numpy(), cmap='jet')
        ax.set_title(f'Theta: {theta:.2f}')
        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # Parameters visual attention
    sizeVM = 42  # Size of the kernel
    r0 = int(0.4 * sizeVM)  # Radius shift from the center (8)
    rho = 0.01 * sizeVM  # Scale coefficient to control arc length (0.1)
    theta = np.pi * 3 / 2  # Angle to control the orientation of the arc
    # thick = 0.0001 * sizeVM  #(0.5)  thickness of the arc
    offsetpxs = int(0.35 * sizeVM)  # (6 for )
    offset = (offsetpxs, offsetpxs)
    num_pyrVM = 5
    strideVM = 1
    # Create Von Mises (VM) filters with specified parameters
    # The angles are generated in radians, ranging from 0 to 2π in steps of π/4
    thetas = np.arange(0, 2 * np.pi, np.pi / 4)

    VMkernels = create_vm_filters(thetas, sizeVM, rho, r0, offset)

    ###### start again from the creation of the filters

    plot_VMkernels(VMkernels, thetas)