import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#tkagg is the backend for matplotlib
matplotlib.use('TkAgg')
# Add a slider to navigate through the maps
from matplotlib.widgets import Slider


def update(val):
    idx = int(slider.val)
    axes[0].imshow(windows[idx])
    axes[1].imshow(attention_maps[idx])
    fig.canvas.draw_idle()


def read_npy_file(file_path):
    """Read and print the contents of an .npy file."""
    try:
        data = np.load(file_path)
        print(f"Contents of {file_path}:\n")
        print(data)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except IOError:
        print(f"Error: An error occurred while reading the file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return data

if __name__ == "__main__":
    # Specify the file path
    attpath = "attention_maps.npy"
    attention_maps = read_npy_file(attpath)
    salmapcoordpath = "salmaps_coordinates.npy"
    salmaps_coordinates = read_npy_file(salmapcoordpath)
    windpath = "windows.npy"
    windows = read_npy_file(windpath)

    # visualise all 200 windows and attention maps
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(left=0.1, bottom=0.25)
    axes[0].imshow(windows[0])
    axes[1].imshow(attention_maps[0])
    # Add a slider to navigate through the maps
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, 'Index', 0, len(windows) - 1, valinit=0, valstep=1)
    slider.on_changed(update)
    plt.show()



    print('end')

