import cv2
import math

import numpy as np
import torchvision

from libs.core.inference import get_all_joint_preds

from utils.vis.util import save_batch_vectormaps


def save_batch_image_with_vectors(batch_image, joints, orients, file_name, expand=1, nrow=8):
    """

    :param batch_image: (b, c, h, w)
    :param joints: (b, j, k=3, 2)
    :param orients: (b, j, k=3, 2)
    :param file_name:
    :param expand:
    :param nrow:
    :return:
    """
    grid = torchvision.utils.make_grid(batch_image, nrow, 0, True)
    np_img = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    np_img = np_img.copy()

    if orients is None:
        print('=> Vector direction unavailable')
        cv2.imwrite(file_name, np_img)
        return

    b_size = batch_image.size(0)

    n_x = min(nrow, b_size)
    n_y = int(math.ceil(float(b_size) / n_x))
    height = int(batch_image.size(2))
    width = int(batch_image.size(3))

    b = 0
    for y in range(n_y):
        for x in range(n_x):
            j = joints[b]
            v = orients[b]
            for joint_list, vec_list in zip(j, v):
                for joint, vec in zip(joint_list, vec_list):
                    joint[0] *= expand
                    joint[1] *= expand
                    joint[0] += x * width
                    joint[1] += y * height
                    px = int(joint[0])
                    py = int(joint[1])
                    vx = int(vec[0] * 1000)
                    vy = int(vec[1] * 1000)
                    cv2.circle(np_img, (px, py), 2, [0, 255, 0], 2)
                    cv2.line(np_img, (px, py), (px + vx, py + vy), (0, 255, 0), 3)
            b += 1

    cv2.imwrite(file_name, np_img)


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

    preds, _ = get_all_joint_preds(batch_heatmaps.detach().cpu().numpy())

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


def save_debug_images(config, input, meta, target, pred_j, pred_v, out_hm, out_vm, prefix):
    """Save result images for debugging

    :param config:
    :param input:
    :param meta:
    :param target:
    :param pred_j:
    :param pred_v:
    :param out_hm:
    :param out_vm:
    :param prefix:
    :return:
    """
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        gt_j = meta['joints_xyv'][:, :, :, 0:2]
        gt_v = meta['joints_xyv'][:, :, :, 2:4]
        gt_o = gt_v - gt_j
        save_batch_image_with_vectors(input, gt_j, gt_o, f'{prefix}_gt.jpg', expand=1)
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_vectors(input, pred_j, pred_v, f'{prefix}_pred.jpg', expand=4)  # 4=384/96
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(input, target, f'{prefix}_hm_gt.jpg')
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(input, out_hm, f'{prefix}_hm_pred.jpg')
        save_batch_vectormaps(input, out_vm, f'{prefix}_vm_pred_.jpg')
