#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import cv2
import random
import numpy as np

from PIL import Image


def normalize_vector_components(cmp):
    tmp = np.zeros(cmp.shape)
    out = np.zeros(cmp.shape)
    tmp[..., 1] = np.sqrt(np.power(cmp[..., 0], 2) + np.power(cmp[..., 1], 2))
    out[..., 0] = cmp[..., 0] / tmp[..., 1]
    out[..., 1] = cmp[..., 1] / tmp[..., 1]
    return out


def crop_image(image, center, size):
    cty, ctx = center
    height, width = size
    im_height, im_width = image.shape[0:2]
    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = height // 2, width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
        cropped_cty - top,
        cropped_cty + bottom,
        cropped_ctx - left,
        cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - height // 2,
        ctx - width // 2
    ])

    return cropped_image, border, offset


def cv_img_to_pil(cv_image):
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil_img_to_cv(pil_image):
    cv_image = np.array(pil_image)
    return cv_image[:, :, ::-1].copy()


def cv_resize_img(cv_image, max_width=720):
    width = cv_image.shape[1]
    if width > max_width:
        width_resized = max_width
    else:
        width_resized = width
    height_resized = int(cv_image.shape[0] * width_resized / width)
    dim = (width_resized, height_resized)
    return cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)


def check_cv_image_status(cv_image):
    """检查给定cv2图像的合法性, 若合法, 反馈其宽度, 高度, 以及通道数

    :param cv_image:
    :return: ok, w, h, c
    """
    if not isinstance(cv_image, np.ndarray):
        return False, None, None, None

    img_shape = cv_image.shape
    if len(img_shape) == 2:
        h = img_shape[0]
        w = img_shape[1]
        c = 1
        if w > 1 and h > 1:
            return True, w, h, c
        else:
            return False, None, None, None
    elif len(img_shape) == 3:
        h = img_shape[0]
        w = img_shape[1]
        c = img_shape[2]
        if w > 1 and h > 1:
            return True, w, h, c
        else:
            return False, None, None, None
    else:
        return False, None, None, None


def get_h_matrix(kp_src, kp_dst, good_match):
    """根据关键点匹配关系, 求解实现由源关键点集到目标关键点集的单应性矩阵, 返回该矩阵

    :param kp_src: 源关键点集
    :param kp_dst: 目标关键点集
    :param good_match: list -> DMatch 匹配关系
    :return:
    """
    pts_src = []
    pts_dst = []
    for g in good_match:
        qid = g.queryIdx  # idx of source key point list
        tid = g.trainIdx  # idx of target key point list
        # pt: (w, h)
        pts_src.append([kp_src[qid].pt[0], kp_src[qid].pt[1]])
        pts_dst.append([kp_dst[tid].pt[0], kp_dst[tid].pt[1]])

    h, _ = cv2.findHomography(np.array(pts_src), np.array(pts_dst), cv2.RANSAC)
    if h is not None:
        return True, h
    else:
        return False, None


def transform_by_h(xy_list, h, dim=2):
    """使用单应性变换矩阵, 将平面上一组点集xy_list变换为另一组点集

    :param xy_list: list -> [[x1, y1], [x2, y2], ...]
    :param h: 3x3 ndarray
    :param dim: 输出列表的维度, dim=2则为 [[x1, y1], [x2, y2],...] dim=1则为[x1, y1, x2, y2, ...]
    :return: xy_list_trans
    """
    if isinstance(xy_list, np.ndarray):
        xy_list = xy_list.tolist()

    xy_list_trans = []
    for xy in xy_list:
        xy_extend = np.array([[xy[0]], [xy[1]], [1]])
        xy_trans = np.matmul(h, xy_extend)
        if dim == 2:
            xy_list_trans.append([float(xy_trans[0]), float(xy_trans[1])])
        else:
            xy_list_trans.append(float(xy_trans[0]))
            xy_list_trans.append(float(xy_trans[1]))
    return xy_list_trans


def align_src_to_dst(img_src, img_dst, h):
    """利用单应性变换矩阵h, 将源图像校正至与目标图像重合, 返回校正后的图像

    :param img_src: ndarray 源图像
    :param img_dst: ndarray 目标图像
    :param h: 3x3 ndarray 单应性矩阵
    :return: ok, img_aligned
    """
    img_shape = img_dst.shape
    try:
        img_aligned = cv2.warpPerspective(img_src, h, (img_shape[1], img_shape[0]))
        return True, img_aligned
    except RuntimeError:
        return False, None


def get_bbox_iou(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    w = x_b - x_a + 1
    h = y_b - y_a + 1
    if w <= 0 or h <= 0:
        return 0
    else:
        inter_area = w * h

        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        return iou


def get_most_overlap_part(bbox, part_info_list):
    """对给定bbox, 在part_info_list中搜索与之最重合的part并返回

    :param bbox: list -> int [x_lt, y_lt, x_rb, y_rb]
    :param part_info_list: list -> dict 部件信息字典中必须包含‘bbox’键
    :return: ok, best_part
    """
    if not part_info_list:
        return False, None
    max_iou = 0
    best_part = None
    for part in part_info_list:
        bbox_dst = part['bbox']
        iou = get_bbox_iou(bbox, bbox_dst)

        if iou > max_iou:
            max_iou = iou
            best_part = part
    if not best_part:
        return False, None
    else:
        return True, best_part


def compute_iou(_rec1, _rec2):
    """
    computing IoU
    :param _rec1: (x0, y0, x1, y1), which reflects
    :param _rec2:
    :return: scala value of IoU
    """
    rec1 = [_rec1[1], _rec1[0], _rec1[3], _rec1[2]]
    rec2 = [_rec2[1], _rec2[0], _rec2[3], _rec2[2]]
    # computing area of each rectangles
    s_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    s_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = s_rec1 + s_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

