#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import math
import time
import torch
import numpy as np
import torchvision
import torchsnooper
import torch.nn as nn
import matplotlib.pyplot as plt
import libs.core.inference as lib_inference

from PIL import ImageFont, Image
from torchvision import datasets, models, transforms


STANDARD_COLORS = [
    'Aqua', 'Chartreuse', 'Aquamarine', 'Chocolate', 'Coral', 'CornflowerBlue',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'BlueViolet', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'Cornsilk', 'Crimson', 'Cyan'
]

NUM_COLORS = len(STANDARD_COLORS)


def show_tensor_data(inputs, title=None):
    """Imshow for Tensor.
        Use:
        inputs, classes = next(iter(dataloaders['train']))
        class_names = image_datasets['train'].classes
        show_tensor_data(inputs, title=[class_names[x] for x in classes])
    """
    inp = torchvision.utils.make_grid(inputs)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def print_info(content):
    print(''.join(['\033[1m\033[94m[]: ', content, '\033[0m']))


def print_warn(content):
    print(''.join(['\033[1m\033[93m[]: ', content, '\033[0m']))


def print_error(content):
    print(''.join(['\033[1m\033[91m[]: ', content, '\033[0m']))


def print_sys_info():
    print_info("- System info -")
    print_info("Python: {}".format(sys.version))
    print_info("PyTorch: {}".format(torch.__version__))
    print_info("GPU number: {}".format(torch.cuda.device_count()))


def print_args(args):
    args_dict = args.__dict__
    print_info("- Configurations -")
    for key in args_dict.keys():
        print_info("{}: {}".format(key, args_dict[key]))


def cv_img_to_pil(cv_image):
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil_img_to_cv(pil_image):
    cv_image = np.array(pil_image)
    return cv_image[:, :, ::-1].copy()


def save_np_heatmaps(batch_heatmaps, file_name, is_max=True):
    """
    batch_heatmaps: should be numpy.ndarray
    file_name: saved file name
    """
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    grid_image = np.zeros((batch_size * heatmap_height, num_joints * heatmap_width, 3), dtype=np.uint8)

    if is_max:
        preds, maxvals = lib_inference.get_max_preds(batch_heatmaps)
    else:
        preds, maxvals = lib_inference.get_all_preds(batch_heatmaps)

    for i in range(batch_size):
        heatmaps = np.clip(batch_heatmaps[i] * 255, 0, 255)

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :].astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            if is_max:
                p = preds[i][j]
                # print(f'preds max {i}, {j}: {p}')
                cv2.circle(colored_heatmap, (int(p[0]), int(p[1])), 1, [0, 255, 0], 1)
            else:
                for p in preds[i][j]:
                    # print(f'preds all {i}, {j}: {p}')
                    cv2.circle(colored_heatmap, (int(p[0]), int(p[1])), 1, [0, 255, 0], 1)

            width_begin = heatmap_width * j
            width_end = heatmap_width * (j + 1)
            grid_image[height_begin:height_end, width_begin:width_end, :] = colored_heatmap

    cv2.imwrite(file_name, grid_image)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True, ratio=0.5):
    """
    :param batch_image: [batch_size, channel, height, width]
    :param batch_heatmaps: ['batch_size, num_joints, height, width]
    :param file_name: saved file name
    :param normalize:
    :param ratio:
    :return:
    """
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
                          dtype=np.uint8)

    preds, maxvals = lib_inference.get_all_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255).clamp(0, 255).byte().cpu().numpy()

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            point_list = preds[i][j]
            for point in point_list:
                cv2.circle(resized_image, (point[0], point[1]), 1, [0, 0, 255], 1)

            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            masked_image = colored_heatmap * (1.0 - ratio) + resized_image * ratio

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def apply_dot(draw, xy_list, w, h, idx=0):
    """
    """
    color = STANDARD_COLORS[idx % NUM_COLORS]
    d = int(min(w, h) * 0.02)

    for xy in xy_list:
        x = xy[0]
        y = xy[1]
        ellipse_bbox = [x - d, y - d, x + d, y + d]
        draw.ellipse(ellipse_bbox, fill=color)


def apply_line(draw, line, w, h, idx=0):
    """
    :param draw: PIL Draw
    :param line: list -> [x, y, x, y]
    :param w: int
    :param h: int
    :param idx: int
    """
    color = STANDARD_COLORS[idx % NUM_COLORS]
    draw.line(line, width=int(min(w, h) * 0.01), fill=color)

