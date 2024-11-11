from Speckegolayer_functions import *
from configSpeckmain import *
from controller.helper import run_controller


if __name__ == "__main__":

    ##### egomotion

    # create kernel Gaussian distribution
    gauss_kernel_center = gaussian_kernel(size_krn_center, sigma_center)
    gauss_kernel_surround = gaussian_kernel(size_krn_surround, sigma_surround)

    # plot_kernel(gauss_kernel_center,gauss_kernel_center.size(0))
    # plot_kernel(gauss_kernel_surround,gauss_kernel_surround.size(0))

    filter = gauss_kernel_surround - gauss_kernel_center
    # plot_kernel(filter, filter.size(0))
    filter = filter.unsqueeze(0)

    # Initialize the network with the loaded filter
    # net_center = net_def(filter_center, tau_mem, num_pyr, size_krn_center)
    # net_surround = net_def(filter_surround, tau_mem, num_pyr, size_krn_surround)
    net = net_def(filter,tau_mem, num_pyr, filter.size(1))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


    ##### attention

    # 'thetas' defines the orientation angles, 'size' defines the filter size, and other parameters shape the filters
    filters_attention = create_vm_filters(thetas, size, rho, r0, thick, offset)

    # Toggle to display the created filters for visualization (currently set to False)
    # plot_filters(filters_attention, thetas)

    # Initialize the attention network with the loaded filters
    netattention = network_init(filters_attention)


    # Set the Speck and sink events
    sink, window, numevs, events_lock = Specksetup()
    # Start the event-fetching thread
    event_thread = threading.Thread(target=fetch_events, args=(sink, window, drop_rate, events_lock, numevs))
    event_thread.daemon = True
    event_thread.start()
    showstats = True
    if showstats:
        plt.figure()
    # Main loop for visualization
    while True:
        current_time = time.time()
        with events_lock:
            if current_time - last_update_time > update_interval:
                if numevs[0] > 0:
                    egomap, indexes = egomotion(window, net, numevs, device)
                    # cv2.imshow('DVS Events', window)
                    egomap = np.array(egomap.detach().cpu().numpy(), dtype=np.uint8)
                    salmap, salmap_coords = run_attention(egomap, netattention)
                    cv2.imshow('Egomotion', egomap[0])
                    cv2.waitKey(1)
                    window.fill(0)
                    # are the location of maximum attention value, values are between -1 and 1
                    cmd = run_controller(
                            np.array([salmap_coords[1]/(resolution[1]), salmap_coords[0]/(resolution[1])]),
                            np.array([0.5, 0.5])
                        )
                    if showstats:
                        #print number of events
                        print('Number of events: ' + str(numevs[0]))
                        print('Number of indexes:', indexes.sum().item())
                        plt.plot([current_time], [numevs[0]], 'ro-', label='Events')
                        plt.plot([current_time], [indexes.sum().item()], 'bo-', label='Events after suppression')
                        plt.plot([current_time], [numevs[0] - indexes.sum().item()], 'yo-', label='Events dropping')
                        plt.title('Comparison of Events and Indexes Over Time')
                        plt.xlabel('Time')
                        plt.ylabel('Events Count')
                        if not plt.gca().get_legend():
                            plt.legend()
                        plt.pause(0.001)  # Pause to update the figure
                    numevs[0] = 0
                last_update_time = current_time
