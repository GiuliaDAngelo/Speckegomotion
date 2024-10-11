from PIL import Image

from egomotionlayer_functions import *
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from config import *


def run(res, filter, egomaps, net, frames, max_x, max_y, alpha):
    cnt=0
    egomap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
    save_egomap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
    incmotionmap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)

    meanrunstats = torch.zeros((max_y, max_x))
    varrunstats = torch.zeros((max_y, max_x))
    sdrunstats = torch.zeros((max_y, max_x))

    for frame in frames:
        print(str(cnt) + "frame out of " + str(frames.shape[0]))
        #scales pyramid
        for pyr in range(1, num_pyr+1):
            Hin = res[pyr-1][0]
            Win = res[pyr-1][1]
            Hw = int(Hin - filter.size()[1] + 1)
            Ww = int(Win - filter.size()[2] + 1)
            # print(f"pyramid scale {pyr}")
            #get resolution and resize input for the pyramid
            frm_rsz = torchvision.transforms.Resize((res[pyr-1][0], res[pyr-1][1]))(frame)
            with torch.no_grad():
                output = net(frm_rsz.float())
            # print(output.shape)
            #normalising over the neurons on the layer
            egomap[0, (pyr - 1)] = torchvision.transforms.Resize((max_y, max_x))(output) / (Hw * Ww)

        # sum egomaps from the pyramid + running stats for running mean
        egomap = torch.sum(egomap, dim=1, keepdim=True)
        # show the egomap plt.draw()
        #put flag to show the egomap
        if show_egomap:
            plt.imshow(egomap[0][0].cpu().detach().numpy(), cmap='jet')
            plt.draw()
            plt.pause(0.0001)
        [meanrunstats, varrunstats,sdrunstats] = running_stats(egomap[0][0], meanrunstats, varrunstats,alpha)

        # egomap - running mean activity
        incmotionmap = egomap[0][0] - meanrunstats
        if show_netactivity:
            plt.imshow(incmotionmap.cpu().detach().numpy(), cmap='jet')
            plt.draw()
            plt.pause(0.0001)
        #save results, normalise only for visualisation
        egomaps[cnt] = egomap
        save_egomap = (egomap - egomap.min()) / (egomap.max() - egomap.min())
        incmotionmap  = (incmotionmap - incmotionmap.min()) / (incmotionmap.max() - incmotionmap.min())

        #save egomap and meanmaps+varmaps+sdmaps
        vutils.save_image(save_egomap.squeeze(0), respath+'/egomaps/egomap'+str(cnt)+'.png')
        vutils.save_image(incmotionmap.squeeze(0), respath + '/incmotionmaps/incmotionmap' + str(cnt) + '.png')
        plt.imsave(respath + '/meanmaps/meanmap'+str(cnt)+'.png', meanrunstats, cmap='jet')
        plt.imsave(respath + '/varmaps/varmap'+str(cnt)+'.png', varrunstats, cmap='jet')
        plt.imsave(respath + '/sdmaps/sdmap'+str(cnt)+'.png', sdrunstats, cmap='jet')

        #empty egomaps
        egomap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
        save_egomap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.float32)
        cnt+=1
    return egomaps


def load_eventsh5(polarity, time_wnd_frames, ds_arr):
    # structured the data
    ds_arr[:, 0] -= int(ds_arr[0][0])  # starting from the first timestep
    rec_data = rf.unstructured_to_structured(ds_arr,
                                             dtype=np.dtype(
                                                 [('t', int), ('x', np.int16), ('y', np.int16), ('p', bool)]))

    # get the sensor size
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
    return frames, max_y, max_x

def createfld(respath, namefld):
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(respath+namefld):
        os.makedirs(respath+namefld)
        print('Folder created')
    else:
        print('Folder already exists')

