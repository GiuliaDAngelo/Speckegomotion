import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')


respath = '/Users/giuliadangelo/workspace/code/Speckegomotion/results/'

dirs_exp = [d for d in os.listdir(respath) if os.path.isdir(os.path.join(respath, d))]
#sort dirs_exp
dirs_exp.sort()
max_x = 488
max_y = 350
dur_video = 2
time_wnd_frames= 0.01834

x = np.arange(len(dirs_exp)*2)
pos = 0
order = [2, 4, 1, 5, 7, 8, 9, 6, 3]

for i in range(1, len(dirs_exp)+1):
    num = order.index(i)
    dir = dirs_exp[num]
    with open(respath+dir+'/spikes.pkl', 'rb') as f:
        spikes = pickle.load(f)

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
    # Plot histogram with error bars
    plt.bar(pos, meanMFRNeurons, yerr=meanstdMFRNeurons, capsize=5, alpha=0.7, color='white', edgecolor='black')
    plt.bar(pos+1, meanISINeurons, yerr=meanstdISINeurons, capsize=5, alpha=0.7, color='lightcoral', edgecolor='black')
    pos+=2


labels = ['   1', '', '    2', '', '   3', '', '   4','', '   5','', '   6','', '   7', '', '   8', '','   9', ''] # Custom x-axis labels
# Customize the plot  # Set custom x-axis labels
plt.xlabel('Experiment',fontsize=18)
# plt.ylabel('Values')
# plt.title('Mean with Standard Deviation as Error Bars')
plt.legend(['MFR [Hz]', 'ISI [s]'], fontsize=18)
plt.xticks(x, labels, fontsize=18)
plt.yticks(fontsize=18) # Set custom x-axis labels
plt.show()

# plt.figure()
# for ind in range(len(spikes)):
#     print(ind)
#     neuron = spikes[ind]
#     plt.vlines(neuron, ymin=ind-0.5, ymax=ind+0.5, color='black', linewidth=1)
# # plt.xlim(0, time_wnd_frames * len_fr)
# plt.ylim(-0.5, len(spikes) - 0.5)
# # plt.xlabel('Time (s)')
# # plt.ylabel('Neuron')
# # plt.title('Raster plot of Spikes')
# plt.show()

# save figure
# plt.savefig(exp+'.png')


print('end')
