import h5py
import numpy as np
import numpy.lib.recfunctions as rf
import tonic
import matplotlib.pyplot as plt
import torchvision
import torch


def h5load_data(filename):
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]  # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        print(ds_arr)
    return data, ds_arr


filename = "data/egomotionstimuli/ego_objmoving/events.h5"
polarity = True
#load data
[data, ds_arr] = h5load_data(filename)
#structured the data
ds_arr[:, 0] -= int(ds_arr[0][0]) #starting from the first timestep
rec_data = rf.unstructured_to_structured(ds_arr,
                                        dtype=np.dtype(
                                            [('t', int), ('x', np.int16), ('y', np.int16), ('p', bool)]))
time_wnd_frames = 505
#get the sensor size
max_x = rec_data['x'].max().astype(int)
max_y = rec_data['y'].max().astype(int)
max_ts = rec_data['t'].max()
# use single polarity
rec_data['p'][rec_data['p'] == False] = True
rec_data['p'] = polarity
sensor_size = (max_x + 1, max_y + 1, 1)
# print(f"sensor size is {sensor_size}")
# convert events into frames
transforms = torchvision.transforms.Compose([
    tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_wnd_frames),
    torch.tensor,
])
frames = transforms(rec_data)

plt.imshow(frames[0][0])
plt.show()


print('end')

