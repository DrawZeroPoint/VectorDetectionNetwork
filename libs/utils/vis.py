from __future__ import absolute_import
from __future__ import division

import math

import numpy as np
import torchvision
import cv2

from libs.core.inference import get_all_preds


def save_batch_image_with_joints(batch_image, pred_j, pred_v, file_name, expand=1, nrow=8, padding=0):
    """

    :param batch_image:
    :param pred_j: joints
    :param pred_v: (k, 2)
    :param file_name:
    :param expand:
    :param nrow:
    :param padding:
    :return:
    """
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    if pred_v is None:
        cv2.imwrite(file_name, ndarr)
        return

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    b = 0

    for y in range(ymaps):
        for x in range(xmaps):
            if b >= nmaps:
                break

            joints = pred_j[b]
            vectors = pred_v[b]

            for joint_list, vec_list in zip(joints, vectors):
                for joint, vec in zip(joint_list, vec_list):
                    joint[0] *= expand
                    joint[1] *= expand
                    joint[0] += x * width + padding
                    joint[1] += y * height + padding
                    px = int(joint[0])
                    py = int(joint[1])
                    vx = int(vec[0] * 100)
                    vy = int(vec[1] * 100)
                    cv2.circle(ndarr, (px, py), 2, [0, 255, 0], 2)
                    cv2.line(ndarr, (px, py), (px + vx, py + vy), (0, 255, 0), 3)
            b += 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True):
    """
    :param batch_image: [batch_size, channel, height, width]
    :param batch_heatmaps: [batch_size, num_joints, height, width]
    :param file_name:
    :param normalize:
    :return:
    """
    if normalize:
        batch_image = batch_image.clone()
        vmin = float(batch_image.min())
        vmax = float(batch_image.max())

        batch_image.add_(-vmin).div_(vmax - vmin + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3), dtype=np.uint8)

    preds, _ = get_all_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255).clamp(0, 255).byte().cpu().numpy()

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)

        for j in range(num_joints):
            point_list = preds[i][j]
            for point in point_list:
                cv2.circle(resized_image, (point[0], point[1]), 1, [0, 255, 0], 1)

            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, pred_j, pred_v, output, prefix):
    if not config.DEBUG.DEBUG:
        return

    # if config.DEBUG.SAVE_BATCH_IMAGES_GT:
    #     save_batch_image_with_joints(input, meta['joints_xyv'], '{}_gt.jpg'.format(prefix), expand=1)
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(input, pred_j, pred_v, '{}_pred.jpg'.format(prefix), expand=4)
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(input, target, '{}_hm_gt.jpg'.format(prefix))
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(input, output, '{}_hm_pred.jpg'.format(prefix))
