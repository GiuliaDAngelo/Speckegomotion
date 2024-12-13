"""
Model developed by:
Giulia D'Angelo
Alexander Hadjiivanov

Summary:
This script loads a Gaussian filter, initializes a spiking neural network, and processes event-based data from a video file.
The results are saved as different types of maps (egomaps, meanmaps, varmaps, etc.) and converted into videos.
The events can be loaded either from .npy or .h5 files, and the filter is applied across frames to extract motion information.

Steps:
1. Load the Gaussian filter.
2. Load event-based data from an .npy file.
3. Run the spiking neural network on the data.
4. Save the generated frames and create videos from the resulting maps.
"""

# --------------------------------------
from egomotion import conf
from egomotion import utils
from egomotion.conf import logger
from egomotion.network import kernels
from egomotion.network import AttentionNet
from egomotion.controllers import Hub
from egomotion.controllers import NengoController
from egomotion.devices import SpeckDevice
from egomotion.devices import PTU


def main_egomotion():

    # REVIEW: Is this still used?
    # create results folders
    if conf.stim_save_res:
        utils.mkdir(conf.respath)

    # create kernel Gaussian distribution
    gauss_kernel = kernels.gaussian(conf.k_size, conf.k_sigma)
    # plot_kernel(gauss_kernel,gauss_kernel.size(0))
    filter = gauss_kernel.unsqueeze(0)

    # Load event-based data from a specified .npy file.
    [frames, max_y, max_x, time_wnd_frames] = utils.load_eventsnpy(
        conf.ev_polarity,
        conf.ev_video_dur,
        conf.ev_video_fps,
        conf.filePathOrName,
        conf.ev_ts_ms,
    )

    # run(filter_dog, filter_gauss, frames, max_x, max_y,time_wnd_frames)
    Hub.run(filter, frames, max_x, max_y, time_wnd_frames)

    # Save the results as videos
    if conf.stim_save_res:
        utils.create_video_from_frames(
            conf.respath + "egomaps/",
            conf.respath + conf.respath.split("/")[1] + "egomap.mp4",
            fps=conf.ev_video_fps,
        )

    print("end")


def analyse_ems():
    """
    Analysis mean over event-frames.

    REVIEW: Do we still need this?
    """

    respath = "results/resIEBCS"
    # load data
    [meanegomap, frames, time_wnd_frames, numframes] = utils.load_files(respath)
    title = "egomotion"

    # plot running mean
    utils.plot_runningmean(numframes, time_wnd_frames, meanegomap, respath, title)

    # create videos from frames
    utils.create_video_from_frames(
        respath + "/egomaps", respath + "/" + title + ".mp4", fps=30
    )
    # create_video_from_tensor(frames, respath+'/'+title+'edframes.mp4', fps=30)


def main():

    out_path = utils.mkdir(conf.OUT_DIR / "results/objego/")
    # respath = 'results/ego/'
    # respath = 'results/onlyobj/'

    # respath_to_file = {
    #     "results/ego/": "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego1/ego1.npy",
    #     "results/ego3/": "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego3/ego3.npy",
    #     "results/ego4/": "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego4/ego4.npy",
    #     "results/ego5/": "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego5/ego5.npy",
    #     "results/ego8/": "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/ego/ego8/ego8.npy",
    #     "results/objego/": "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/objego/objego.npy",
    #     "results/onlyobj/": "/Users/giuliadangelo/workspace/code/IEBCS/data/video/egomotionstimuli/onlyobj/onlyobj.npy",
    # }

    # Kernels
    # ==================================================
    logger.info("Creating the kernels...")
    center_kernel = kernels.gaussian(
        conf.k_center_size,
        conf.k_center_sigma,
    )
    surround_kernel = kernels.gaussian(
        conf.k_surround_size,
        conf.k_surround_sigma,
    )
    attention_kernel = kernels.make_vm_filters(
        conf.vm_thetas,
        conf.vm_size,
        conf.vm_rho,
        conf.vm_r0,
        conf.vm_thick,
        conf.vm_offset,
    )

    # Egomition / attention network
    # ==================================================
    logger.info("Setting up the attention network...")
    net = AttentionNet(
        center_kernel,
        surround_kernel,
        attention_kernel,
        conf.tau_mem,
        conf.vm_num_pyr,
    )

    # The Nengo controller
    # ==================================================
    logger.info("Setting up the Nengo controller...")
    controller = NengoController()

    # The PTU
    # ==================================================
    logger.info("Setting up the PTU interface...")
    ptu = PTU()

    # The Speck device
    # ==================================================
    logger.info("Setting up the camera...")
    speck = SpeckDevice()

    # The hub
    # ==================================================
    hub = Hub(net, ptu, speck, controller)

    hub.run()

if __name__ == "__main__":
    main()
