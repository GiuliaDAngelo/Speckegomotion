import numpy as np
import cv2
import torch
from functions.attention_helpers import AttentionModule
import os
from functions.OMS_helpers import initialize_oms, egomotion
import matplotlib.pyplot as plt


def process_llevos_events(directory, dir, file):
    evdata = np.load(directory + dir + '/' + file, allow_pickle=True)
    ev = np.zeros(len(evdata), dtype=[('x', 'i2'), ('y', 'i2'), ('t', 'f8'), ('p', 'b')])
    ev['x'] = evdata[:, 0].astype(int)
    ev['y'] = evdata[:, 1].astype(int)
    ev['t'] = evdata[:, 2]
    ev['t'] = (ev['t'] - ev['t'][0])
    ev['p'] = evdata[:, 3].astype(bool)
    max_y = ev['y'].max() + 1
    max_x = ev['x'].max() + 1
    return ev, max_y, max_x


def mkdirfold(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Folder created')
    else:
        print('Folder already exists')

if __name__ == '__main__':

    directory = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO2LowightChallengingConditions/'
    events_dir = directory+ 'events/'
    annotations_dir = directory + 'annotations/'
    ev_frames_dir = directory + 'ev_frames/'
    oms_dir = directory + 'oms/'
    attention_dir = directory + 'attention/'


    dirs_events = [d for d in os.listdir(events_dir) if os.path.isdir(os.path.join(events_dir, d))]
    dirs_events = sorted(dirs_events)


    for dir in dirs_events:
        ev_files = [f for f in os.listdir(events_dir + dir) if f.endswith('.npy')]
        ev_files = sorted(ev_files)
        print(dir)
        cnt = 1
        pos_dir = ev_frames_dir+dir+'/pos/'
        neg_dir = ev_frames_dir + dir + '/neg/'
        mkdirfold(pos_dir)
        mkdirfold(neg_dir)

        for ev_file in ev_files:
            inner_cnt = 0
            ev, max_y, max_x = process_llevos_events(events_dir, dir, ev_file)
            window_pos = torch.zeros((max_y, max_x), dtype=torch.uint8)
            window_neg = torch.zeros((max_y, max_x), dtype=torch.uint8)

            wnd = 50000
            t = wnd
            ev['t'] = ev['t']  - ev['t'][0]
            windows_pos = []
            windows_neg = []

            for x, y, ts, p in ev:
                if ts<=t:
                    if p:
                        window_pos[y, x] = 255
                    else:
                        window_neg[y, x] = 255
                else:
                    windows_pos.append(window_pos)
                    windows_neg.append(window_neg)
                    # cv2.imshow('Event pos map', window_pos.cpu().numpy())
                    # cv2.imshow('Event neg map', window_neg.cpu().numpy())

                    plt.imsave(pos_dir + f'/{cnt:05}_{inner_cnt}.png', window_pos, cmap='gray')
                    plt.imsave(neg_dir + f'/{cnt:05}_{inner_cnt}.png', window_neg, cmap='gray')

                    cv2.waitKey(1)
                    window_pos = torch.zeros((max_y, max_x), dtype=torch.uint8)
                    window_neg = torch.zeros((max_y, max_x), dtype=torch.uint8)
                    t+=wnd
                    inner_cnt+=1
            cnt+=1