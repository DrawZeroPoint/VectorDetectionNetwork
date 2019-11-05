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
from libs.core.inference import get_max_preds
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


def visualize_classification(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                show_tensor_data(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


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
    print(''.join(['\033[1m\033[94m[LingXi]: ', content, '\033[0m']))


def print_warn(content):
    print(''.join(['\033[1m\033[93m[LingXi]: ', content, '\033[0m']))


def print_error(content):
    print(''.join(['\033[1m\033[91m[LingXi]: ', content, '\033[0m']))


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

    grid_image = np.zeros((batch_size * heatmap_height, (num_joints) * heatmap_width, 3), dtype=np.uint8)

    if is_max:
        preds, maxvals = lib_inference.get_max_preds(batch_heatmaps)
    else:
        preds, maxvals = lib_inference.get_all_preds(batch_heatmaps)

    for i in range(batch_size):
        heatmaps = np.clip(batch_heatmaps[i] * 255, 0 ,255)

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :].astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap
            width_begin = heatmap_width * j
            width_end = heatmap_width * (j + 1)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

    cv2.imwrite(file_name, grid_image)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True, ratio=0.5):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
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

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255).clamp(0, 255).byte().cpu().numpy()

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            # cv2.circle(resized_image, (int(preds[i][j][0]), int(preds[i][j][1])), 1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * (1.0 - ratio) + resized_image * ratio

            # cv2.circle(masked_image, (int(preds[i][j][0]), int(preds[i][j][1])), 1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def apply_dot(draw, xy, w, h, idx=0):
    """
    """
    color = STANDARD_COLORS[idx % NUM_COLORS]
    d = int(min(w, h) * 0.02)
    ellipse_bbox = [xy[0] - d, xy[1] - d, xy[0] + d, xy[1] + d]
    draw.ellipse(ellipse_bbox, fill=color)


def apply_line(draw, line, w, h, idx=0):
    """在PIL画板上绘制线段

    :param draw: PIL Draw
    :param line: list -> [x, y, x, y]
    :param w: int 图像宽度
    :param h: int 图像高度
    :param idx: int 标签序号, 决定线条颜色
    """
    color = STANDARD_COLORS[idx % NUM_COLORS]
    draw.line(line, width=int(min(w, h) * 0.01), fill=color)


def apply_frame(draw, bbox, w, h, idx=0):
    """在PIL画板上绘制矩形框

    :param draw: PIL Draw
    :param bbox: list -> int 矩形框 (x_lt, y_lt, x_rb, y_rb)
    :param w: int 图像宽度
    :param h: int 图像高度
    :param idx: int 标签序号, 决定矩形框颜色, 默认为0
    """
    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    color = STANDARD_COLORS[idx % NUM_COLORS]
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=int(min(w, h) * 0.01), fill=color)


def apply_text(draw, pt, text, w, h, idx=0, smart_offset=True, bg = False):
    """在PIL画板上绘制文字

    :param draw: PIL Draw
    :param pt: List -> Int 绘制参考点[x, y], 为文字左下角点
    :param text: String 需要添加的文字
    :param w: Int 图像宽度
    :param h: Int 图像高度
    :param idx: int 标签序号, 决定文字颜色
    :param smart_offset: 当文字超出图像范围时智能移位
    """
    min_dim = min(w, h)
    font_sz = int(min_dim * 0.08)
    offset_x = int(w * 0.02)
    offset_y = int(h * 0.02)+0.1
    bg_w = math.ceil(font_sz * 3)
    bg_h = math.ceil(font_sz * 1.2)
    font = ImageFont.truetype("/LingXi/fonts/NotoSansCJKsc-Regular.otf", font_sz)
    color = STANDARD_COLORS[idx]
    color_b = STANDARD_COLORS[81]
    # 绘制文字背景
    if smart_offset:
        if pt[0] + offset_x + bg_w > w:
            offset_x = -(bg_w + offset_x)
        if pt[1] + offset_y + font_sz > h:
            offset_y = -(font_sz + offset_y)
    if bg:
        draw.rectangle(((pt[0] + offset_x, pt[1] + offset_y),
                        (pt[0] + offset_x + bg_w, pt[1] + offset_y + bg_h)), fill=color_b)
    # 绘制文字
    draw.text((pt[0] + offset_x, pt[1] + offset_y - offset_y / abs(offset_y) * 2),
              text, font=font, fill=color)


def apply_cross(draw, pt, length=10, color=(255, 255, 255)):
    """在PIL画板上绘制十字

    :param draw: PIL.Draw
    :param pt: List [x, y] Cross center point
    :param length: 十字半边臂长
    :param color: Tuple Color of the cross
    """
    draw.line((pt[0] - length, pt[1], pt[0] + length + 1, pt[1]), color, width=2)
    draw.line((pt[0], pt[1] - length, pt[0], pt[1] + length + 1), color, width=2)


def concat_image_list(image_list, one_row=True):
    """将图像列表中的图像按行或列拼接为单一图像返回

    :param image_list: list -> ndarray -> BGR 图像列表
    :param one_row: bool 若为真则将图像拼接为一行, 否则为一列
    :return: ok, concat_image
    """
    if not image_list:
        return None

    row_sep = (np.ones((10, 400, 3)) * 255).astype(np.uint8)
    col_sep = (np.ones((400, 10, 3)) * 255).astype(np.uint8)

    for i, img in enumerate(image_list):
        img_r = cv2.resize(img, (400, 400))
        if i == 0:
            concat_image = img_r
        else:
            if one_row:
                concat_image = np.concatenate([concat_image, col_sep, img_r], axis=1)
            else:
                concat_image = np.concatenate([concat_image, row_sep, img_r], axis=0)
    return concat_image
