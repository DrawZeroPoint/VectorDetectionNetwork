#!/usr/bin/env python
# encoding: utf-8

import math
import torchsnooper
import numpy as np

from scipy.ndimage.measurements import label
from skimage.measure import regionprops

from libs.utils.transforms import transform_preds


# @torchsnooper.snoop()
def get_all_preds(batch_heatmaps):
    """Get predictions (n, c, 2), 2 for (x, y), from model output

    :param batch_heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-dim'

    batch_size = batch_heatmaps.shape[0]  # 2
    num_joints = batch_heatmaps.shape[1]  # 5

    preds = []
    maxvals = []
    for n in range(batch_size):
        joints_peaks = []
        joints_maxvals = []
        for j in range(num_joints):
            heatmap = batch_heatmaps[n, j, :]
            bw = heatmap
            bw[bw < 0.6] = 0
            labeled_bw, _ = label(bw)
            cc = regionprops(labeled_bw)

            peaks_in_one_map = []
            peak_vals_in_one = []
            cnt = 0
            for k, c in enumerate(cc):
                centroid = c.centroid  # tuple
                row = int(centroid[0] + 0.5)
                col = int(centroid[1] + 0.5)
                peak = [col, row]
                peaks_in_one_map.append(peak)
                peak_val = heatmap[row, col]
                peak_vals_in_one.append(peak_val)
                cnt = k + 1

            peaks = np.array(peaks_in_one_map)
            peakvals = np.array(peak_vals_in_one)
            print(f'Found {cnt} contours in batch {n}, joint {j}; peaks {peaks}, peak_vals {peakvals}')

            joints_peaks.append(peaks)
            joints_maxvals.append(peakvals)

        preds.append(joints_peaks)
        maxvals.append(joints_maxvals)

    return np.array(preds), np.array(maxvals)


def get_max_preds(batch_heatmaps):
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]  # 2
    num_joints = batch_heatmaps.shape[1]  # 5
    width = batch_heatmaps.shape[3]  # input_width / 4
    # np.save("/VDN/data/test.npy", batch_heatmaps)
    # raise ValueError()
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    
    idx = np.argmax(heatmaps_reshaped, 2)  # Returns the indices of the maximum values along an axis
    maxvals = np.amax(heatmaps_reshaped, 2)  # Return the maximum of an array or maximum along an axis.

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # Given predicted pixel idx, get its coordinate (x, y) as preds
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)  # Turn true of false to float

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    # coords, maxvals = get_max_preds(batch_heatmaps)
    coords, maxvals = get_all_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                     hm[py + 1][px] - hm[py - 1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals
