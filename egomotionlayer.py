from PIL import Image
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pickle
from egomotionlayer_functions import *



def run(res, filter, egomaps, net, frames, max_x, max_y, num_pyr, respath,alpha):
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
        [meanrunstats, varrunstats,sdrunstats] = running_stats(egomap[0][0], meanrunstats, varrunstats,alpha)

        # egomap - running mean activity
        incmotionmap = egomap[0][0] - meanrunstats
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


if __name__ == '__main__':
    #load DoG filter
    path_data = "dog_kernel.npy"
    filter = load_krn(path_data)

    #instantiate network
    net = net_def(filter)


    ###################################
    ###########LOAD EVENTS#############
    ###################################

    #load events npy
    # polarity = True
    # time_wnd_frames = 5000  # us
    # # filePathOrName = "data/objclutter.npy"
    # filePathOrName = "data/objclutter.npy"
    # # respath = "results"
    # respath = "results/resegoobj2"
    # [frames,max_y, max_x]=load_eventsnpy(polarity, time_wnd_frames, filePathOrName, respath)

    #load events h5
    # polarity = True
    # filename = "data/egomotionstimuli/egomotion1/events.h5"
    # respath =  "results/egomotion"
    # [data, ds_arr] = h5load_data(filename)
    # time_wnd_frames = ds_arr[0][0]
    # [frames, max_y, max_x] = load_eventsh5(polarity, time_wnd_frames, ds_arr)

    filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego_objmoving/ego_objmoving.npy"
    # filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/onlyego/egomotion1/egomotion1.npy"
    # filePathOrName = "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/onlyobj/onlyobj.npy"

    # respath = "results/onlyobj"
    respath = "results/objego"
    # respath = "results/ego"
    # load data
    title = respath.split('/')[1]

    #load events IEBCS
    polarity = True
    FPSvideo = 60.0
    dur_video=4#seconds
    tsFLAG=False #flag to converst the timestamps in microseconds

    [frames, max_y, max_x,time_wnd_frames] = load_eventsnpy(polarity, dur_video, FPSvideo, filePathOrName, tsFLAG)
    stimspeed=30.0 #px/s
    deltaT=time_wnd_frames
    tau=time_wnd_frames*4
    alpha=np.exp(-deltaT/tau)

    ###################################
    ###################################
    ###################################

    #create folders for egomaps + meanmaps, varmaps and sdmaps
    createfld(respath, '/egomaps')
    createfld(respath, '/meanmaps')
    createfld(respath, '/varmaps')
    createfld(respath, '/sdmaps')
    createfld(respath,'/incmotionmaps')

    #run network
    num_pyr = 1
    res = pyr_res(num_pyr, frames) # get the resolutions for the pyramid
    egomaps = torch.empty((frames.shape[0], 1, max_y, max_x), dtype=torch.int64)
    numframes=len(frames)
    frames = frames[1:numframes, :, :, :]
    egomaps = run(res, filter, egomaps, net, frames, max_x, max_y, num_pyr, respath, alpha)

    #save data
    with open(respath+'/frames.pkl', 'wb') as f:
        pickle.dump(frames, f)
    with open(respath+'/time_wnd_frames.pkl', 'wb') as f:
        pickle.dump(time_wnd_frames, f)
    with open(respath+'/numframes.pkl', 'wb') as f:
        pickle.dump(numframes, f)

######## Analysis results

    #create videos from frames
    create_video_from_frames(respath+'/egomaps', respath+'/'+title+'.mp4', fps=FPSvideo)
    create_video_from_frames(respath + '/meanmaps', respath + '/' + title + 'mean.mp4', fps=FPSvideo)
    create_video_from_frames(respath + '/varmaps', respath + '/' + title + 'var.mp4', fps=FPSvideo)
    create_video_from_frames(respath + '/sdmaps', respath + '/' + title + 'sd.mp4', fps=FPSvideo)
    create_video_from_frames(respath + '/incmotionmaps', respath + '/' + title + 'incmotion.mp4', fps=FPSvideo)


    print('end')
