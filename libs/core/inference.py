#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import torchsnooper
import numpy as np
import scipy.ndimage as ndimage

from scipy.ndimage.measurements import label
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage import img_as_float

from libs.utils.transforms import transform_preds


def get_peaks_by_regions(heatmap):
    bw = heatmap
    bw[bw < np.max(heatmap) * 0.9] = 0
    labeled_bw, _ = label(bw)
    cc = regionprops(labeled_bw)

    peaks = []
    peakvals = []
    for c in cc:
        centroid = c.centroid  # tuple
        row = int(centroid[0] + 0.5)
        col = int(centroid[1] + 0.5)

        peak = np.array([col, row])  # x, y
        peaks.append(peak)
        peak_val = heatmap[row, col]
        peakvals.append(peak_val)

    return peaks, peakvals


def get_peaks_by_local_maximum(heatmap):
    heatmap = img_as_float(heatmap)

    """
    High pass filter on heatmap
    min_distance for 2 adjacent points to be reduced as 1, because the heatmap is small,
    we should have low min_distance to prevent points near the borders getting lost
    """
    coordinates = peak_local_max(heatmap, min_distance=10, threshold_rel=0.2)
    if not np.any(coordinates) == 0:
        print('------> get_peaks_by_local_maximum: peaks is empty')

    peaks = []
    peakvals = []
    for c in coordinates:
        row = int(c[0] + 0.5)
        col = int(c[1] + 0.5)

        peak = np.array([col, row])
        peaks.append(peak)
        peak_val = heatmap[row, col]
        peakvals.append(peak_val)

    return peaks, peakvals


# @torchsnooper.snoop()
def get_all_joint_preds(batch_heatmaps):
    """Get predictions (n, c, k, 2), 2 for (x, y), from model output

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

            # Get all peaks (local maximum) within the heatmap
            peaks, peakvals = get_peaks_by_regions(heatmap)

            joints_peaks.append(peaks)
            joints_maxvals.append(peakvals)

        preds.append(joints_peaks)
        maxvals.append(joints_maxvals)

    return np.array(preds), maxvals


def get_all_orientation_preds(pred_all_joints, vector_maps):
    """

    :param pred_all_joints: (b, j, k, 2), Notice that for the last dim, 2 values are x and y, not h and w
    :param vector_maps: (b, 2, h, w)
    :return: ndarray (b, j, k, 2)
    """
    vector_map_width = vector_maps.shape[3]
    vector_maps = np.reshape(vector_maps, (vector_maps.shape[0], vector_maps.shape[1], -1))

    if len(pred_all_joints.shape) == 4 and pred_all_joints.shape[-1] != 0:
        x = pred_all_joints[:, :, :, 0]
        y = pred_all_joints[:, :, :, 1]
        idx = x + y * vector_map_width
        preds_idx = np.squeeze(idx, 1)  # (b, k)
    elif len(pred_all_joints.shape) == 3 and pred_all_joints.shape[-1] != 0:
        x = pred_all_joints[:, :, 0]
        y = pred_all_joints[:, :, 1]
        idx = x + y * vector_map_width
        preds_idx = idx  # (b, k)
    else:
        return None

    vectormaps_t = torch.from_numpy(vector_maps)  # (b, 2, 96*96)
    preds_idx_t = torch.from_numpy(preds_idx)  # (b, k)

    preds_idx_t = preds_idx_t.expand(preds_idx_t.size(0), 2, preds_idx_t.size(1))  # (b, 2, k)
    # print(f'preds_idx_t {preds_idx_t.size()} {preds_idx_t} vectormap_t {vectormaps_t.size()}')

    preds_vector_t = vectormaps_t.gather(2, preds_idx_t)  # (b, 2, k)
    preds_vector = preds_vector_t.permute(0, 2, 1).unsqueeze(1).numpy()  # (b, j=1, k, 2)

    return preds_vector


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


# @torchsnooper.snoop()
def get_final_preds(batch_heatmaps, batch_vectormaps, center, scale):
    batch_sz = batch_heatmaps.shape[0]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    preds_j, maxvals = get_all_joint_preds(batch_heatmaps)
    preds_v = get_all_orientation_preds(preds_j, batch_vectormaps)

    # Transform back
    for i in range(batch_sz):
        preds_j[i] = transform_preds(preds_j[i], center[i], scale[i], [heatmap_width, heatmap_height])
        if preds_v is not None:
            preds_v[i] = transform_preds(preds_v[i], center[i], scale[i], [heatmap_width, heatmap_height])

    return preds_j, preds_v, maxvals
