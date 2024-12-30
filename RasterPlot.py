import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')


# respath = '/Users/giuliadangelo/workspace/code/Speckegomotion/results/resultsegoobjegoonlyobj/'
respath = '/Users/giuliadangelo/workspace/code/Speckegomotion/results/objegokernel/'



dirs_exp = [d for d in os.listdir(respath) if os.path.isdir(os.path.join(respath, d))]
#sort dirs_exp
dirs_exp.sort()
max_x = 488
max_y = 350
dur_video = 2
time_wnd_frames= 0.01834


pos = 0
# order = [2, 4, 1, 5, 7, 8, 9, 6, 3] #resultsegoobjegoonlyobj

# order = [2, 3, 6, 7, 4, 5, 8, 0, 1] #objegokernel
order = [2, 3, 6, 7, 5, 8] #objegokernel sigma
# order = [2, 0, 1] #objegokernel kernel size



x = np.arange(len(order)*2)
plt.figure(figsize=(10, 8))
for i in range(0, len(order)):
    # num = order.index(i)
    # dir = dirs_exp[num]

    dir = dirs_exp[order[i]]
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


fontsize = 18
# labels = ['   1', '', '    2', '', '   3', '', '   4','', '   5','', '   6','', '   7', '', '   8', '','   9', '']
# labels = ['   sc1ss4krn8', '', '    sc2ss4krn8', '', '   sc3ss4krn8', '', '   sc4ss4krn8','', '   sc2ss6krn8','', '   sc2ss8krn8','', '   sc4ss8krn8', '', '   sc4ss8krn16', '','   sc8ss16krn32', ''] # Custom x-axis labels
labels = ['     Σc=1,Σs=4', ' ', '      Σc=2,Σs=4', ' ', '      Σc=3,Σs=4', ' ', '     Σc=4,Σs=4',' ', '     Σc=2,Σs=8',' ', '     Σc=4,Σs=8', ' '] # Custom x-axis labels
# labels = ['            Σc=2,Σs=4; s=8', ' ', '            Σc=4,Σs=8; s=16', ' ', '            Σc=8,Σs=16; s=32', ' '] # Custom x-axis labels


# Customize the plot  # Set custom x-axis labels
# plt.xlabel('Experiment',fontsize=fontsize)
# plt.ylabel('Values')
# plt.title('Mean with Standard Deviation as Error Bars')
plt.legend(['MFR [Hz]', 'ISI [s]'], fontsize=fontsize)
plt.xticks(x, labels, fontsize=fontsize)
plt.yticks(fontsize=fontsize) # Set custom x-axis labels
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
