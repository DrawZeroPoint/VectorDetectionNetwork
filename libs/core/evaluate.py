#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np

import libs.core.inference as lib_inference


def dist_acc(dists, thr=0.5):
    """Return percentage below threshold while ignoring values with a -1

    """
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


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
    # print(f'pred_joints {pred_joints.shape}, target {targets.shape}')

    norm = np.ones(2) * np.array([h, w]) / 10
    dists = np.zeros((num_joints, batch_sz))
    for n in range(batch_sz):
        for c in range(num_joints):
            preds_list = pred_joints[n][c]
            targets_list = gt_joints[n][c]
            dist_all_pred = 0
            for pred in preds_list:
                dist_one_pred = 0
                for target in targets_list:
                    if target[0] > 1 and target[1] > 1:
                        normed_preds = pred / norm
                        normed_targets = target / norm
                        dist = np.linalg.norm(normed_preds - normed_targets)
                    else:
                        dist = -1
                    dist_one_pred += dist
                dist_all_pred += dist_one_pred
            dists[n, c] = dist_all_pred

    # print(f'pred_joints {pred_joints.shape}, target {targets.shape}')

    pred_vectors = lib_inference.get_all_orientation_preds(pred_joints, vectormaps)
    gt_vectors = lib_inference.get_all_orientation_preds(gt_joints, target_vectormaps)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred_joints, pred_vectors, gt_vectors
