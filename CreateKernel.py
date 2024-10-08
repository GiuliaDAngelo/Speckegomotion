import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

# Function to create a Difference of Gaussian kernel in PyTorch
def difference_of_gaussian(size, sigma1, sigma2):
    # Create a grid of (x, y) coordinates using PyTorch
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y)

    # Create two Gaussian kernels with different sigmas
    gaussian1 = (torch.exp(-(x**2 + y**2) / (2 * sigma1**2)))/(np.sqrt(2*np.pi)*sigma1)

    gaussian2 = (torch.exp(-(x**2 + y**2) / (2 * sigma2**2)))/(np.sqrt(2*np.pi)*sigma2)

    # Calculate Difference of Gaussian (DoG)
    dog = gaussian1 - gaussian2

    # Normalize the DoG values between 0 and 1
    # dog_normalized = (dog - dog.min()) / (dog.max() - dog.min())

    return dog

# Parameters
size = 25  # Size of the kernel (NxN)
sigma1 = 0.04*size  # Sigma for the first Gaussian
sigma2 = 2*size  # Sigma for the second Gaussian

# Generate DoG kernel as a tensor
dog_kernel = difference_of_gaussian(size, sigma1, sigma2)

# Convert the tensor to NumPy array for 3D plotting
dog_kernel_np = dog_kernel.numpy()

# Plot 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for 3D plotting
x = torch.linspace(-size // 2, size // 2, size).numpy()
y = torch.linspace(-size // 2, size // 2, size).numpy()
x, y = np.meshgrid(x, y)

# Plot the surface
ax.plot_surface(x, y, dog_kernel_np, cmap='viridis')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Amplitude')
ax.set_title('Difference of Gaussian Kernel (PyTorch)')

plt.show()
np.save('dog_kernel.npy', dog_kernel_np)


print('end')
