import h5py
import numpy as np
import numpy.lib.recfunctions as rf
import tonic
import torch
import torchvision
import pickle


def convert_h5_to_npy():

    filename = "data/egomotionstimuli/ego_objmoving/events.h5"
    polarity = True
    # load data
    [data, ds_arr] = load_h5_data(filename)
    # structured the data
    ds_arr[:, 0] -= int(ds_arr[0][0])  # starting from the first timestep
    rec_data = rf.unstructured_to_structured(
        ds_arr,
        dtype=np.dtype([("t", int), ("x", np.int16), ("y", np.int16), ("p", bool)]),
    )
    time_wnd_frames = 505
    # get the sensor size
    max_x = rec_data["x"].max().astype(int)
    max_y = rec_data["y"].max().astype(int)
    max_ts = rec_data["t"].max()
    # use single polarity
    rec_data["p"][rec_data["p"] == False] = True
    rec_data["p"] = polarity
    sensor_size = (max_x + 1, max_y + 1, 1)
    # print(f"sensor size is {sensor_size}")
    # convert events into frames
    transforms = torchvision.transforms.Compose(
        [
            tonic.transforms.ToFrame(
                sensor_size=sensor_size, time_window=time_wnd_frames
            ),
            torch.tensor,
        ]
    )
    frames = transforms(rec_data)

    # plt.imshow(frames[0][0])
    # plt.show()


def load_h5_data(filename):
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        # print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]  # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        # print(ds_arr)
    return data, ds_arr


def load_eventsh5(polarity, time_wnd_frames, ds_arr):
    # structured the data
    ds_arr[:, 0] -= int(ds_arr[0][0])  # starting from the first timestep
    rec_data = rf.unstructured_to_structured(
        ds_arr,
        dtype=np.dtype([("t", int), ("x", np.int16), ("y", np.int16), ("p", bool)]),
    )

    # get the sensor size
    max_x = rec_data["x"].max().astype(int)
    max_y = rec_data["y"].max().astype(int)
    max_ts = rec_data["t"].max()
    # use single polarity
    rec_data["p"][rec_data["p"] == False] = True
    rec_data["p"] = polarity
    sensor_size = (max_x + 1, max_y + 1, 1)
    # print(f"sensor size is {sensor_size}")
    # convert events into frames
    transforms = torchvision.transforms.Compose(
        [
            tonic.transforms.ToFrame(
                sensor_size=sensor_size, time_window=time_wnd_frames
            ),
            torch.tensor,
        ]
    )
    frames = transforms(rec_data)
    return frames, max_y, max_x


def npy_data(filePathOrName, tsFLAG):
    recording = np.load(filePathOrName)
    if tsFLAG:
        recording[:, 3] *= 1e3  # convert time from seconds to milliseconds
    rec = rf.unstructured_to_structured(
        recording,
        dtype=np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", int)]),
    )
    return rec


def load_eventsnpy(polarity, dur_video, FPS, filePathOrName, tsFLAG):
    rec = npy_data(filePathOrName, tsFLAG)
    # find out maximum x and y

    ### values here are in milliseonds the max ts should be the duration of the video 4 sec
    max_x = rec["x"].max().astype(int)
    max_y = rec["y"].max().astype(int)
    max_ts = rec["t"].max()
    # use single polarity
    rec["p"][rec["p"] == False] = True
    rec["p"] = polarity
    sensor_size = (max_x + 1, max_y + 1, 1)
    time_wnd_frames = rec["t"].max() / dur_video / FPS
    # print(f"sensor size is {sensor_size}")
    # We have to convert the raw events into frames so that we can feed those to our network
    # We use a library called tonic for that https://tonic.readthedocs.io/en/latest/ as well as torchvision
    transforms = torchvision.transforms.Compose(
        [
            tonic.transforms.ToFrame(
                sensor_size=sensor_size, time_window=time_wnd_frames
            ),
            torch.tensor,
        ]
    )
    frames = transforms(rec)
    return frames, max_y, max_x, time_wnd_frames


def load_files(respath):
    with open(respath + "/meanegomap.pkl", "rb") as f:
        meanegomap = pickle.load(f)
    with open(respath + "/frames.pkl", "rb") as f:
        frames = pickle.load(f)
    with open(respath + "/time_wnd_frames.pkl", "rb") as f:
        time_wnd_frames = pickle.load(f)
    with open(respath + "/numframes.pkl", "rb") as f:
        numframes = pickle.load(f)
    return meanegomap, frames, time_wnd_frames, numframes
