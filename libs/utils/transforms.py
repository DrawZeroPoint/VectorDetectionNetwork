#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
import torchsnooper


# @torchsnooper.snoop()
def transform_preds(coords, center, scale, output_size):
    # print(f'coords {coords}')
    # [list([array([46, 46]), array([44, 50]), array([45, 52])])
    #  list([array([45, 44])]) list([array([75, 88])])]

    target_coords = []
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for k in range(coords.shape[0]):
        point_list = coords[k]
        target_list = []
        for p in point_list:
            pt = affine_transform(p, trans)
            target_list.append(pt)
        target_coords.append(target_list)
    target_coords = np.array(target_coords)

    return target_coords


def get_affine_transform(center, shape, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    """
    """
    shape_expanded = shape * 200.0
    src_w = shape_expanded[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + shape_expanded * shift
    src[1, :] = center + src_dir + shape_expanded * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img
