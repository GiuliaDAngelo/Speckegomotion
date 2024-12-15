import numpy as np
import torchvision
import tonic
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import sinabs.layers as sl
import pickle

import matplotlib
matplotlib.use('TkAgg')



def loadeventsEVIMO(evpath, dir, npz, file):
    ###### ###### ###### ###### ###### ######
    ###### convert events into frames ######
    ###### ###### ###### ###### ###### ######
    # timestamp, x, y, p
    evdata = np.load(evpath + dir + npz + file, allow_pickle=True)
    events = evdata['events']
    ev = np.zeros(len(events), dtype=[('t', 'f8'), ('x', 'i2'), ('y', 'i2'), ('p', 'b')])
    ev['t'] = events[:, 0]
    ev['x'] = events[:, 1].astype(int)
    ev['y'] = events[:, 2].astype(int)
    ev['p'] = np.ones_like(events[:, 3], dtype=bool)
    max_x = ev['x'].max() + 1
    max_y = ev['y'].max() + 1

    index = evdata['index']
    # K = evdata['K']
    # D = evdata['D']
    # depth = evdata['depth']
    mask = evdata['mask']
    meta = evdata['meta']
    GT = meta.item()['frames']

    discretization = evdata['discretization']
    sensor_size = (max_x, max_y, 1)
    time_wnd_frames = discretization

    # Convert dictionary to structured NumPy array
    # structured_ev = np.zeros(len(ev['t']), dtype=[('t', 'i8'), ('x', 'i2'), ('y', 'i2'), ('p', 'b')])
    transforms = torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_wnd_frames),
        torch.tensor,
    ])
    evframes = transforms(ev)
    return evframes, max_y, max_x, mask, GT, time_wnd_frames



def savekernel(kernel, size, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = torch.linspace(0, size, size)
    y = torch.linspace(0, size, size)
    x, y = torch.meshgrid(x, y)
    ax.plot_surface(x.numpy(), y.numpy(), kernel[0].numpy(), cmap='jet')
    plt.show()
    plt.savefig(name+'.png')


def OMSkernels():
    # create kernel Gaussian distribution
    center = gaussian_kernel(size_krn_center, sigma_center).unsqueeze(0)
    surround = gaussian_kernel(size_krn_surround, sigma_surround).unsqueeze(0)
    return center, surround

def gaussian_kernel(size, sigma):
    # Create a grid of (x, y) coordinates using PyTorch
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y, indexing='ij')  # Ensure proper indexing for 2D arrays
    # Create a Gaussian kernel
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # Normalize the kernel so that the values are between 0 and 1
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    return kernel

def net_def(filter, tau_mem, num_pyr, size_krn, device, stride):
    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(1, num_pyr, (size_krn,size_krn), padding = 'same', stride=stride, bias=False),
        sl.LIF(tau_mem),
    )
    net[0].weight.data = filter.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net


def egomotion(window, net_center, net_surround, device, max_y, max_x,threshold):
    window = window.unsqueeze(0).float().to(device)
    center = net_center(window)
    surround = net_surround(window)
    # center = torch.nn.functional.interpolate(center.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
    #                                          align_corners=False).squeeze(0)
    # surround = torch.nn.functional.interpolate(surround.unsqueeze(0), size=(max_y, max_x), mode='bilinear',
    #                                          align_corners=False).squeeze(0)

    events = center - surround
    events = 1 - (events - events.min())/(events.max() - events.min())
    indexes = events >= threshold

    if indexes.any():
        OMS = torch.zeros_like(events)
        OMS[indexes] = 255
    else:
        OMS = torch.zeros_like(events)

    # center = (center - center.min()) / (center.max() - center.min())
    # surround = (surround - surround.min()) / (surround.max() - surround.min())
    # center = center * 255
    # surround = surround * 255
    # events = events * 255
    # fig, axs = plt.subplots(1, 4, figsize=(15, 10))
    # axs[0].cla()
    # axs[1].cla()
    # axs[2].cla()
    # axs[3].cla()
    # axs[0].imshow(center[0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
    # axs[1].imshow(surround[0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
    # axs[2].imshow(events[0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
    # axs[3].imshow(OMS[0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=255)
    # plt.draw()
    # plt.pause(0.001)
    return OMS, indexes

def mkdirfold(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Folder created')
    else:
        print('Folder already exists')

def getIOU(spike_pred, spike_gt):
    spike_pred = spike_pred.cpu().numpy()
    spike_gt = spike_gt.cpu().numpy()

    intersection = np.sum(np.logical_and(spike_pred, spike_gt))
    union = np.sum(np.logical_or(spike_pred, spike_gt))
    return intersection/union




# Load the .npz file
evimofld = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/'
evpath = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/EVIMOevents/'
maskpath = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/EVIMOmasks/'
evfrpath = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/EVIMOevframes/'
maskfrpath = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/EVIMOmasksframes/'
OMSpath = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO/OMS/'

# look at dirs whitin the path
dirs_events = [d for d in os.listdir(evpath) if os.path.isdir(os.path.join(evpath, d))]
dirs = [d for d in os.listdir(evfrpath) if os.path.isdir(os.path.join(evfrpath, d))]


# Parameters kernel
size_krn_center = 8  # Size of the kernel (NxN) (all half ) - 8
sigma_center = 2  # Sigma for the first Gaussian - 1
size_krn_surround = 8  # Size of the kernel (NxN) - 8
sigma_surround = 4  # Sigma for the first Gaussian - 4

eventsFLAG = False
showimages = False
tau_mem = 0.02 #(events 0.03)
threshold = 0.80 #(events 0.60)
num_pyr = 1
maxBackgroundRatio = 3.5



if eventsFLAG:
    for dir in dirs_events:
        npz = '/npz/'
        #look at files in the dir
        files = [f for f in os.listdir(evpath+dir+npz) if f.endswith('.npz')]
        #sort list of files
        files = sorted(files)
        for file in files:
            seq_name = file.split('.')[0]
            maskfile = seq_name +'_masks.npy'

            evmaskdata = np.load(maskpath + dir + '/' + maskfile, allow_pickle=True)
            [evframes, max_y, max_x, mask, GT, time_wnd_frames] = loadeventsEVIMO(evpath, dir, npz, file)
            evmaskdata = np.load(maskpath+dir+'/'+maskfile, allow_pickle=True)

            #results folder
            res_path = dir+'/'+seq_name+'/'
            print(res_path)
            evframeres = evfrpath + res_path
            maskframeres = maskfrpath + res_path
            OMSframeres = OMSpath + res_path

            mkdirfold(evframeres)
            mkdirfold(maskframeres)
            mkdirfold(OMSframeres)

            ###### ###### ###### ###### ###### ######
            ###### ###### prepare network ###### ####
            ###### ###### ###### ###### ###### ######

            center, surround = OMSkernels()
            # savekernel(center, size_krn_center, 'center')
            # savekernel(surround, size_krn_surround, 'surround')
            # savekernel((center-surround), size_krn_surround, 'OMSkernel')

            ss = 1
            sc = ss + sigma_surround - sigma_center

            # Initialize the network with the loaded filter
            # define our single layer network and load the filters
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

            ###### ###### ###### ###### ###### ######
            ###### ###### run experiments ###### ####
            ###### ###### ###### ###### ###### ######

            time = 0
            i = 0
            # fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            IOUs = []
            timestamps = [gt['ts'] for gt in GT]
            for evframe in evframes:
                if time >= timestamps[i]:
                    net_center = net_def(center, tau_mem, num_pyr, size_krn_center, device, sc)
                    net_surround = net_def(surround, tau_mem, num_pyr, size_krn_surround, device, ss)

                    curr_frame = (evframe[0] != 0.00).clone().detach().to(torch.int)
                    OMS, indexes = egomotion(curr_frame, net_center, net_surround, device, max_y, max_x, threshold)

                    ###################################
                    ############ IoU ##################
                    ###################################
                    spk_mask = torch.tensor(mask[i] != 0.00, dtype=torch.bool)
                    spk_evframe = curr_frame

                    spike_gt = torch.zeros_like(spk_mask)
                    torch.logical_and(spk_mask, spk_evframe, out=spike_gt)

                    objspikes = torch.sum(spike_gt)
                    back_spikes = torch.sum(spk_evframe) - objspikes

                    if back_spikes / objspikes < maxBackgroundRatio:
                        # axs[0].imshow(evframe[0])
                        # axs[1].imshow(masked_spike_tensor.cpu().detach().numpy())
                        # axs[2].imshow(OMS.squeeze(0).cpu().detach().numpy(), vmin=0, vmax=255)
                        # axs[3].imshow(background_spikes.cpu().detach().numpy())

                        spike_pred = torch.zeros_like(OMS[0])
                        spk_oms = (OMS[0] != 0.00).clone().detach().to(torch.int)
                        torch.logical_and(spk_oms, spk_evframe.to(device), out=spike_pred)

                        IOUframe = getIOU(spike_pred, spike_gt)
                        IOUs.append(IOUframe)
                        print('IoU: ' + str(IOUframe))

                        indexes_gt = evmaskdata[i] != 0
                        ground_truth = np.zeros_like(evmaskdata[i], dtype=evframe[0].dtype)
                        ground_truth[indexes_gt] = evframe[0][indexes_gt]

                        # axs[0].imshow(evframe[0])
                        # axs[1].imshow(ground_truth)
                        # axs[2].imshow(spike_gt.cpu().numpy())

                        plt.imsave(evframeres + f'evframe_{i}.png', evframe[0], cmap='gray')
                        plt.imsave(maskframeres + f'mask_{i}.png', spike_gt.cpu().numpy(), cmap='gray')
                        plt.imsave(OMSframeres + f'OMS_{i}.png', spike_pred.cpu().numpy(), cmap='gray')
                        plt.draw()
                        plt.pause(0.001)
                        # print('frame: ' + str(i) + ' out of ' + str(len(timestamps)))
                    i += 1
                time += time_wnd_frames
            print('IOU: ' + str(np.mean(IOUs)))
            with open(OMSpath + dir + '/' +seq_name+'IOUs.pkl', 'wb') as f:
                pickle.dump(IOUs, f)
            with open(OMSpath + dir + '/' +seq_name+ 'meanIOUs.pkl', 'wb') as f:
                pickle.dump(IOUs, f)

else:
    for dir in dirs:
        if dir=='wall' or dir=='tabletop' or dir=='fast' or dir=='box' or dir=='table':
            continue
        # look at files in the dir
        files = [f for f in os.listdir(evfrpath + dir) if f.endswith('.npy')]
        # sort list of files
        files = sorted(files)
        for file in files:
            seq_name = file.split('_')[0]+'_'+file.split('_')[1]
            maskfile = seq_name + '_masks.npy'
            res_path = dir + '/' + seq_name + '/'

            print(res_path)
            evframeres = evfrpath + res_path
            maskframeres = maskfrpath + res_path
            OMSframeres = OMSpath + res_path

            mkdirfold(evframeres)
            mkdirfold(maskframeres)
            mkdirfold(OMSframeres)

            evframesdata = np.load(evfrpath + dir + '/' + file, allow_pickle=True)
            evmaskdata = np.load(maskpath + dir + '/' + maskfile, allow_pickle=True)

            # results folder
            res_path = dir + '/' + seq_name + '/'
            print(res_path)
            OMSframeres = OMSpath + res_path
            mkdirfold(OMSframeres)

            ###### ###### ###### ###### ###### ######
            ###### ###### prepare network ###### ####
            ###### ###### ###### ###### ###### ######

            center, surround = OMSkernels()
            # savekernel(center, size_krn_center, 'center')
            # savekernel(surround, size_krn_surround, 'surround')
            # savekernel((center-surround), size_krn_surround, 'OMSkernel')

            ss = 1
            sc = 1 #original code ss + sigma_surround - sigma_center

            # Initialize the network with the loaded filter
            # define our single layer network and load the filters
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            net_center = net_def(center, tau_mem, num_pyr, size_krn_center, device, sc)
            net_surround = net_def(surround, tau_mem, num_pyr, size_krn_surround, device, ss)

            ###### ###### ###### ###### ###### ######
            ###### ###### run experiments ###### ####
            ###### ###### ###### ###### ###### ######

            i = 0
            if showimages:
                fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            IOUs = []
            max_x = evframesdata[0].shape[2]
            max_y = evframesdata[0].shape[1]
            for evframe in evframesdata:
                print(str(i)+' out of: '+str(len(evframesdata)))
                OMS, indexes = egomotion(torch.tensor(evframe[0]), net_center, net_surround, device, max_y, max_x, threshold)
                ###################################
                ############ IoU ##################
                ###################################
                dens_mask = torch.tensor(evmaskdata[i] != 0.00, dtype=torch.bool)
                spk_evframe = torch.tensor(evframe[0] != 0.00, dtype=torch.bool)
                spk_oms = OMS[0].clone().detach().to(device).bool()

                spike_pred = torch.zeros_like(OMS[0]).bool()
                torch.logical_and(spk_oms, spk_evframe.to(device), out=spike_pred)
                spk_mask = torch.zeros_like(dens_mask).to(device)
                torch.logical_and(dens_mask.to(device),spk_evframe.to(device), out=spk_mask)

                spike_gt = torch.zeros_like(spike_pred)
                torch.logical_and(spike_pred.to(device), dens_mask.to(device), out=spike_gt)

                num_evs_mask = torch.sum(spk_mask).item()
                num_evs_back = torch.sum(spk_evframe).item() - num_evs_mask
                try:
                    ratio = num_evs_back / num_evs_mask
                except ZeroDivisionError:
                    ratio = float('inf')  # or any other value that makes sense in your context

                # print(num_evs_back / num_evs_mask)
                if ratio < maxBackgroundRatio:
                    IOUframe = getIOU(spike_pred, spike_gt)
                    IOUs.append(IOUframe)
                    print('IoU: ' + str(IOUframe))

                    ###################################
                    ############ Plot #################
                    ###################################
                    if showimages:
                        axs[0].cla()
                        axs[1].cla()
                        axs[2].cla()

                    indexes_gt = evmaskdata[i] != 0
                    ground_truth = np.zeros_like(evmaskdata[i], dtype=evframe[0].dtype)
                    ground_truth[indexes_gt] = evframe[0][indexes_gt]

                    if showimages:
                        axs[0].imshow(evframe[0])
                        axs[1].imshow(ground_truth)
                        axs[2].imshow(spike_gt.cpu().numpy())

                    plt.imsave(evframeres + f'evframe_{i}.png', evframe[0], cmap='gray')
                    plt.imsave(maskframeres + f'mask_{i}.png', spike_gt.cpu().numpy(),cmap='gray')
                    plt.imsave(OMSframeres + f'OMS_{i}.png', spike_pred.cpu().numpy(), cmap='gray')
                if showimages:
                    plt.draw()
                    plt.pause(0.001)
                i += 1
            with open(OMSpath + dir + '/' + seq_name + 'IOUs.pkl', 'wb') as f:
                pickle.dump(IOUs, f)
            with open(OMSpath + dir + '/' + seq_name + 'meanIOUs.pkl', 'wb') as f:
                pickle.dump(IOUs, f)



    print('end')