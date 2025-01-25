import cv2
import numpy as np
import torch
import time

from functions.OMS_helpers import *
from functions.attention_helpers import AttentionModule
from functions.Speck_helpers import Specksetup

class Config:
    RESOLUTION = [128, 128]  # Resolution of the DVS sensor
    MAX_X = RESOLUTION[0]
    MAX_Y = RESOLUTION[1]
    DROP_RATE = 0  # Percentage of events to drop
    UPDATE_INTERVAL = 0.001  # seconds
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.96,
        'tau_memOMS': 0.1,
        'sc': 1,
        'ss': 1
    }

    ATTENTION_PARAMS = {
        'VM_radius': 8,  # (R0)
        'VM_radius_group': 15,
        'num_ori': 4,
        'b_inh': 3,  # (w)
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2,  # (rho)
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.3
    }

def compute_OMS(window_pos):
    OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

    OMSpos_map, indexes_pos = egomotion(OMSpos, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
                                        config.OMS_PARAMS['threshold'])

    OMSpos_map = OMSpos_map.squeeze(0).squeeze(0).cpu().detach().numpy()
    return OMSpos_map, indexes_pos

def draw_graph_with_dots(events, suppressed, dropped, width=640, height=480):
    graph_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Define plot areas
    max_events = max(events + suppressed + dropped, default=1)
    margin = 50
    scale_x = (width - 2 * margin) / len(events) if events else 1
    scale_y = (height - 2 * margin) / max_events

    # Draw axes
    cv2.line(graph_img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)  # X-axis
    cv2.line(graph_img, (margin, height - margin), (margin, margin), (0, 0, 0), 2)  # Y-axis

    # Add axis labels
    cv2.putText(graph_img, "Time [s]", (width // 2 - 60, height - margin // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(graph_img, "Event Count", (margin // 2 + 50, height // 6 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # Draw ticks on the X-axis
    num_ticks_x = 5
    for i in range(num_ticks_x + 1):
        x_value = (current_time - start_time) * (i / num_ticks_x)  # Scale time to fit ticks
        x_pos = margin + int(i * (width - 2 * margin) / num_ticks_x)
        cv2.line(graph_img, (x_pos, height - margin - 5), (x_pos, height - margin + 5), (0, 0, 0), 1)
        cv2.putText(graph_img, f"{x_value:.2f}", (x_pos - 20, height - margin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw ticks on the Y-axis
    num_ticks_y = 5
    for i in range(num_ticks_y + 1):
        y_value = int(max_events * (i / num_ticks_y))
        y_pos = height - margin - int(i * (height - 2 * margin) / num_ticks_y)
        cv2.line(graph_img, (margin - 5, y_pos), (margin + 5, y_pos), (0, 0, 0), 1)
        cv2.putText(graph_img, f"{y_value}", (margin - 40, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Plot dots
    for i in range(len(events)):
        x = margin + int(i * scale_x)
        # Total events (red dots)
        y_events = height - margin - int(events[i] * scale_y)
        cv2.circle(graph_img, (x, y_events), 4, (0, 0, 255), -1)

        # Suppressed events (blue dots)
        y_suppressed = height - margin - int(suppressed[i] * scale_y)
        cv2.circle(graph_img, (x, y_suppressed), 4, (255, 0, 0), -1)

    # Add legend
    legend_start_x = width - 200  # Start X position of the legend
    legend_start_y = margin // 2  # Start Y position of the legend
    legend_spacing = 30  # Space between legend entries

    # Legend: Total Events
    cv2.circle(graph_img, (legend_start_x, legend_start_y), 6, (0, 0, 255), -1)
    cv2.putText(graph_img, "Events", (legend_start_x + 20, legend_start_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # Legend: Suppressed Events
    cv2.circle(graph_img, (legend_start_x, legend_start_y + legend_spacing), 6, (255, 0, 0), -1)
    cv2.putText(graph_img, "OMS Events", (legend_start_x + 20, legend_start_y + legend_spacing + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return graph_img


def convert_to_rgb(image):
    # Convert grayscale to RGB by repeating the grayscale values across 3 channels
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

if __name__ == '__main__':
    config = Config()

    net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)
    net_attention = AttentionModule(**config.ATTENTION_PARAMS)
    sink, window_pos, window_neg, numevs, events_lock = Specksetup(config.RESOLUTION, config.DROP_RATE)


    # Scaling factor for enlarging the entire visualization
    scale_factor = 2
    last_update_time = time.time()
    events = []
    suppressed = []
    dropped = []
    start_time = time.time()
    while True:
        current_time = time.time()
        print(current_time - start_time)
        if current_time - start_time > 20:
            print(f"mean of events: {np.round(np.mean(events), 2)}")
            print(f"mean of suppressed events: {np.round(np.mean(suppressed), 2)}")
            break

        with events_lock:
            if current_time - last_update_time > config.UPDATE_INTERVAL:
                if numevs[0] > 0:
                    OMS, indexes = compute_OMS(window_pos)
                    events.append(numevs[0])
                    suppressed.append(indexes.sum().item())
                    dropped.append(numevs[0] - indexes.sum().item())

                    # Generate graph using OpenCV
                    graph_img = draw_graph_with_dots(events, suppressed, dropped)

                    # Get the size of the largest image (graph)
                    max_height = max(window_pos.shape[0], OMS.shape[0], graph_img.shape[0])
                    max_width = max(window_pos.shape[1], OMS.shape[1], graph_img.shape[1])

                    # Scale dimensions for resizing
                    scaled_height = int(max_height * scale_factor)
                    scaled_width = int(max_width * scale_factor)

                    # Create a white background with scaled dimensions
                    background = np.ones((scaled_height, scaled_width * 3, 3), dtype=np.uint8) * 255

                    # Resize each component to scaled dimensions
                    window_pos_resized = convert_to_rgb(cv2.resize(np.flipud(window_pos), (scaled_width, scaled_height)))
                    OMS_resized = convert_to_rgb(cv2.resize(np.flipud(OMS), (scaled_width, scaled_height)))
                    graph_img_resized = cv2.resize(graph_img, (scaled_width, scaled_height))

                    # Place the resized components side by side
                    background[:, :scaled_width] = window_pos_resized
                    background[:, scaled_width:scaled_width * 2] = OMS_resized
                    background[:, scaled_width * 2:] = graph_img_resized

                    cv2.putText(background, 'Event map', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 2)
                    cv2.putText(background, 'OMS map', (max_width + 700, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 2)


                    # Display the updated background
                    cv2.imshow("Visualization", background)
                    cv2.waitKey(1)

                    # Reset data for next frame
                    window_pos.fill(0)
                    window_neg.fill(0)
                    numevs[0] = 0

