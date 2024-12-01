import pickle
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

# Load the spikes_matrix.pkl file
# name_exp = 'objego'
name_exp = 'ego'
# name_exp = 'onlyobj'

with open(name_exp+'.pkl', 'rb') as f:
    spikes = pickle.load(f)


max_x = 257
max_y = 159

dur_video = 4 #seconds
len_fr = 240
time_wnd_frames= dur_video/len_fr

# Example usage of the loaded spikes data
plt.figure()
for ind in range(len(spikes)):
    print(ind)
    neuron  = spikes[ind]
    plt.vlines(neuron, ymin=ind - 2, ymax=ind + 2, color='black',
               linewidth=0.8)
    plt.xlim(0, time_wnd_frames * len_fr)
plt.ylim(-0.5, ind - 0.5)
plt.xlabel('Time (s)')
plt.ylabel('Neuron')
# plt.title('Raster plot of Spikes')
plt.show()

# save figure
# plt.savefig(name_exp+'.png')



print('end')
