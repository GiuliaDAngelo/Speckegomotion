from Speckegolayer_functions import *
from configSpeckmain import *


if __name__ == "__main__":

    # create kernel Gaussian distribution
    gauss_kernel = gaussian_kernel(size_krn, sigma)
    # plot_kernel(gauss_kernel,gauss_kernel.size(0))
    filter = gauss_kernel.unsqueeze(0)

    # Initialize the network with the loaded filter
    net = net_def(filter, tau_mem, num_pyr)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Set the Speck and sink events
    sink, window, numevs, events_lock = Specksetup()
    # Start the event-fetching thread
    event_thread = threading.Thread(target=fetch_events, args=(sink, window, drop_rate, events_lock, numevs))
    event_thread.daemon = True
    event_thread.start()

    # Main loop for visualization
    while True:
        current_time = time.time()
        with events_lock:
            if current_time - last_update_time > update_interval:
                if numevs[0] > 0:
                    suppmap = egomotion(window, net, numevs, device)
                    # cv2.imshow('DVS Events', window)
                    suppmap = np.array(suppmap.cpu())
                    cv2.imshow('Egomotion', suppmap[0])
                    cv2.waitKey(1)
                    window.fill(0)
                    numevs[0] = 0
                last_update_time = current_time