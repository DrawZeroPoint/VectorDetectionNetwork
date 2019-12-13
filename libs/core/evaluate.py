#!/usr/bin/env python
# encoding: utf-8

import scipy
import numpy as np

import libs.core.inference as lib_inference


def dist_acc(dists, thr=0.5):
    """Return percentage below threshold while ignoring values with a -1

    :param dists: [j, b]
    :param thr:
    """
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def get_dist(pred_list, tgt_list, norm):
    # Calculate the cost matrix
    pred_len = len(pred_list)
    gt_len = len(tgt_list)
    pair_dists = np.zeros((pred_len, gt_len))

    for row, pred in enumerate(pred_list):
        for col, target in enumerate(tgt_list):
            normed_pred_joint = pred / norm
            normed_gt_joint = target / norm
            dist = np.linalg.norm(normed_pred_joint - normed_gt_joint)
            pair_dists[row, col] = dist

    # Get the lowest cost with hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(pair_dists)
    dist_all_pred = pair_dists[row_ind, col_ind].sum()

    return dist_all_pred


def accuracy(heatmaps, vectormaps, target_heatmaps, target_vectormaps):
    """
    :param heatmaps: (b, j, h, w)
    :param vectormaps: (b, 2, h, w)
    :param target_heatmaps: (b, j, h, w)
    :param target_vectormaps: (b, 2, h, w)
    :return: pred_joints (b, j, k, 2), 2 for x y; pred_vectors (b, k, 2), 2 for vx vy
    """
    batch_sz = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    h = heatmaps.shape[2]
    w = heatmaps.shape[3]
    idx = list(range(num_joints))

    pred_joints, _ = lib_inference.get_all_joint_preds(heatmaps)
    gt_joints, _ = lib_inference.get_all_joint_preds(target_heatmaps)

    pred_vectors = lib_inference.get_all_orientation_preds(pred_joints, vectormaps)
    gt_vectors = lib_inference.get_all_orientation_preds(gt_joints, target_vectormaps)
    # print(f'pred_joints {pred_joints.shape}, target {targets.shape}')

    norm_j = np.ones(2) * np.array([h, w]) / 10
    norm_v = np.ones(2)
    dists_j = np.ones((batch_sz, num_joints)) * -1.
    dists_v = np.ones((batch_sz, num_joints)) * -1.

    for b in range(batch_sz):
        for n in range(num_joints):
            pred_joint_list = pred_joints[b][n]
            gt_joint_list = gt_joints[b][n]
            dists_j[b, n] = get_dist(pred_joint_list, gt_joint_list, norm_j)

            if pred_vectors is not None and gt_vectors is not None:
                pred_vector_list = pred_vectors[b][n]
                gt_vector_list = gt_vectors[b][n]
                dists_v[b, n] = get_dist(pred_vector_list, gt_vector_list, norm_v)

    # print(f'pred_joints {pred_joints.shape}, target {targets.shape}')

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = (dist_acc(dists_j[idx[i]]) + dist_acc(dists_v[idx[i]])) * 0.5
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred_joints, pred_vectors
