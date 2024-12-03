import pickle
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

# Load the spikes_matrix.pkl file
# name_exp = 'objego'
# name_exp = 'ego'
# name_exp = 'onlyobj'

name_exp = '02'

with open(name_exp+'.pkl', 'rb') as f:
    spikes = pickle.load(f)


max_x = 257
max_y = 159

dur_video = 2#4 #seconds
len_fr = 274#252
time_wnd_frames= dur_video/len_fr

# Example usage of the loaded spikes data
fmeanPerNeuron = np.zeros(len(spikes))
fstdPerNeuron = np.zeros(len(spikes))
MFRneuron = np.zeros(len(spikes))
for ind, neuron in enumerate(spikes):
    # neuron is not empty
    if len(neuron) != 0:
        isi = np.array(neuron)[1:] - np.array(neuron)[0:-1]
        if len(isi) != 0:
            fmeanPerNeuron[ind] = 1/np.mean(isi)
            fstdPerNeuron[ind] = 1/np.std(isi)
            MFRneuron[ind] = len(neuron)/dur_video

#mean ISI of neurons
meanISINeurons = np.mean(fmeanPerNeuron)
meanstdISINeurons = np.mean(fstdPerNeuron)
if meanstdISINeurons == np.inf:
    meanstdISINeurons = 0

meanMFRNeurons = np.mean(MFRneuron)
meanstdMFRNeurons = np.std(MFRneuron)

x = [0, 1]  # Bar positions
labels = ['MFR', 'ISI']  # Custom x-axis labels

# Plot histogram with error bars
plt.bar(x[0], meanMFRNeurons, yerr=meanstdMFRNeurons, capsize=5, alpha=0.7, label='MFR [Hz]', color='skyblue', edgecolor='black')
plt.bar(x[1], meanISINeurons, yerr=meanstdISINeurons, capsize=5, alpha=0.7, label='ISI [s]', color='lightcoral', edgecolor='black')

# Customize the plot
plt.xticks(x, labels)  # Set custom x-axis labels
# plt.xlabel('Metrics')
# plt.ylabel('Values')
# plt.title('Mean with Standard Deviation as Error Bars')
plt.legend(fontsize=18)
plt.xticks(x, labels, fontsize=18)
plt.yticks(fontsize=18) # Set custom x-axis labels
plt.show()

plt.figure()
for ind in range(len(spikes)):
    print(ind)
    neuron = spikes[ind]
    plt.vlines(neuron, ymin=ind-0.5, ymax=ind+0.5, color='black', linewidth=1)
# plt.xlim(0, time_wnd_frames * len_fr)
plt.ylim(-0.5, len(spikes) - 0.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Neuron')
# plt.title('Raster plot of Spikes')
plt.show()

# save figure
# plt.savefig(name_exp+'.png')



print('end')
