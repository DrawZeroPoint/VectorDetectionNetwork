#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 图像处理工具

import cv2
import random
import numpy as np

from PIL import Image


class SiftWrapper(object):
    """"OpenCV SIFT wrapper."""

    def __init__(self, nfeatures=0, n_octave_layers=3,
                 peak_thld=0.0067, edge_thld=10, sigma=1.6,
                 n_sample=8192, patch_size=32):
        self.sift = None

        self.nfeatures = nfeatures
        self.n_octave_layers = n_octave_layers
        self.peak_thld = peak_thld
        self.edge_thld = edge_thld
        self.sigma = sigma
        self.n_sample = n_sample
        self.down_octave = True

        self.sift_init_sigma = 0.5
        self.sift_descr_scl_fctr = 3.
        self.sift_descr_width = 4

        self.first_octave = None
        self.max_octave = None
        self.pyr = None

        self.patch_size = patch_size
        self.standardize = True
        self.output_gird = None

    def create(self):
        """Create OpenCV SIFT detector."""
        self.sift = cv2.xfeatures2d.SIFT_create(
            self.nfeatures, self.n_octave_layers, self.peak_thld, self.edge_thld, self.sigma)

    def detect(self, gray_img):
        """Detect keypoints in the gray-scale image.
        Args:
            gray_img: The input gray-scale image.
        Returns:
            npy_kpts: (n_kpts, 6) Keypoints represented as NumPy array.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
        """

        cv_kpts = self.sift.detect(gray_img, None)

        all_octaves = [np.int8(i.octave & 0xFF) for i in cv_kpts]
        self.first_octave = int(np.min(all_octaves))
        self.max_octave = int(np.max(all_octaves))

        npy_kpts, cv_kpts = sample_by_octave(cv_kpts, self.n_sample, self.down_octave)
        return npy_kpts, cv_kpts

    def compute(self, img, cv_kpts):
        """Compute SIFT descriptions on given keypoints.
        Args:
            img: The input image, can be either color or gray-scale.
            cv_kpts: A list of cv2.KeyPoint.
        Returns:
            sift_desc: (n_kpts, 128) SIFT descriptions.
        """

        _, sift_desc = self.sift.compute(img, cv_kpts)
        return sift_desc

    def build_pyramid(self, gray_img):
        """Build pyramid. It would be more efficient to use the pyramid 
        constructed in the detection step.
        Args:
            gray_img: Input gray-scale image.
        Returns:
            pyr: A list of gaussian blurred images (gaussian scale space).
        """

        gray_img = gray_img.astype(np.float32)
        n_octaves = self.max_octave - self.first_octave + 1
        # create initial image.
        if self.first_octave < 0:
            sig_diff = np.sqrt(np.maximum(
                np.square(self.sigma) - np.square(self.sift_init_sigma) * 4, 0.01))
            base = cv2.resize(gray_img, (gray_img.shape[1] * 2, gray_img.shape[0] * 2),
                              interpolation=cv2.INTER_LINEAR)
            base = cv2.GaussianBlur(base, None, sig_diff)
        else:
            sig_diff = np.sqrt(np.maximum(np.square(self.sigma) -
                                          np.square(self.sift_init_sigma), 0.01))
            base = cv2.GaussianBlur(gray_img, None, sig_diff)
        # compute gaussian kernels.
        sig = np.zeros((self.n_octave_layers + 3,))
        self.pyr = [None] * (n_octaves * (self.n_octave_layers + 3))
        sig[0] = self.sigma
        k = np.power(2, 1. / self.n_octave_layers)
        for i in range(1, self.n_octave_layers + 3):
            sig_prev = np.power(k, i - 1) * self.sigma
            sig_total = sig_prev * k
            sig[i] = np.sqrt(sig_total * sig_total - sig_prev * sig_prev)
        # construct gaussian scale space.
        for o in range(0, n_octaves):
            for i in range(0, self.n_octave_layers + 3):
                if o == 0 and i == 0:
                    dst = base
                elif i == 0:
                    src = self.pyr[(o - 1) * (self.n_octave_layers + 3) + self.n_octave_layers]
                    dst = cv2.resize(
                        src, (int(src.shape[1] / 2), int(src.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
                else:
                    src = self.pyr[o * (self.n_octave_layers + 3) + i - 1]
                    dst = cv2.GaussianBlur(src, None, sig[i])
                self.pyr[o * (self.n_octave_layers + 3) + i] = dst

    def unpack_octave(self, kpt):
        """Get scale coefficients of a keypoints.
        Args:
            kpt: A keypoint object represented as cv2.KeyPoint.
        Returns:
            octave: The octave index.
            layer: The level index.
            scale: The sampling step.
        """

        octave = kpt.octave & 255
        layer = (kpt.octave >> 8) & 255
        octave = octave if octave < 128 else (-128 | octave)
        scale = 1. / (1 << octave) if octave >= 0 else float(1 << -octave)
        return octave, layer, scale

    def get_interest_region(self, scale_img, cv_kpts, standardize=True):
        """Get the interest region around a keypoint.
        Args:
            scale_img: DoG image in the scale space.
            cv_kpts: A list of OpenCV keypoints.
            standardize: (True by default) Whether to standardize patches as network inputs.
        Returns:
            Nothing.
        """
        batch_input_grid = []
        all_patches = []
        bs = 30  # limited by OpenCV remap implementation
        for idx, cv_kpt in enumerate(cv_kpts):
            # preprocess
            _, _, scale = self.unpack_octave(cv_kpt)
            size = cv_kpt.size * scale * 0.5
            ptf = (cv_kpt.pt[0] * scale, cv_kpt.pt[1] * scale)
            ori = (360. - cv_kpt.angle) * (np.pi / 180.)
            radius = np.round(self.sift_descr_scl_fctr * size * np.sqrt(2)
                              * (self.sift_descr_width + 1) * 0.5)
            radius = np.minimum(radius, np.sqrt(np.sum(np.square(scale_img.shape))))
            # construct affine transformation matrix.
            affine_mat = np.zeros((3, 2), dtype=np.float32)
            m_cos = np.cos(ori) * radius
            m_sin = np.sin(ori) * radius
            affine_mat[0, 0] = m_cos
            affine_mat[1, 0] = m_sin
            affine_mat[2, 0] = ptf[0]
            affine_mat[0, 1] = -m_sin
            affine_mat[1, 1] = m_cos
            affine_mat[2, 1] = ptf[1]
            # get input grid.
            input_grid = np.matmul(self.output_grid, affine_mat)
            input_grid = np.reshape(input_grid, (-1, 1, 2))
            batch_input_grid.append(input_grid)

            if len(batch_input_grid) != 0 and len(batch_input_grid) % bs == 0 or idx == len(cv_kpts) - 1:
                # sample image pixels.
                batch_input_grid_ = np.concatenate(batch_input_grid, axis=0)
                patches = cv2.remap(scale_img.astype(np.float32), batch_input_grid_,
                                    None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                patches = np.reshape(patches, (len(batch_input_grid),
                                               self.patch_size, self.patch_size))
                # standardize patches.
                if standardize:
                    patches = (patches - np.mean(patches, axis=(1, 2), keepdims=True)) / \
                              (np.std(patches, axis=(1, 2), keepdims=True) + 1e-8)
                all_patches.append(patches)
                batch_input_grid = []
        if len(all_patches) != 0:
            all_patches = np.concatenate(all_patches, axis=0)
        else:
            all_patches = None
        return all_patches

    def get_patches(self, cv_kpts):
        """Get all patches around given keypoints.
        Args:
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
        Return:
            all_patches: (n_kpts, 32, 32) Cropped patches.
        """

        # generate sampling grids.
        n_pixel = np.square(self.patch_size)
        self.output_grid = np.zeros((n_pixel, 3), dtype=np.float32)
        for i in range(n_pixel):
            self.output_grid[i, 0] = (i % self.patch_size) * 1. / self.patch_size * 2 - 1
            self.output_grid[i, 1] = (i / self.patch_size) * 1. / self.patch_size * 2 - 1
            self.output_grid[i, 2] = 1

        scale_index = [[] for i in range(len(self.pyr))]
        for idx, val in enumerate(cv_kpts):
            octave, layer, _ = self.unpack_octave(val)
            scale_val = (int(octave) - self.first_octave) * (self.n_octave_layers + 3) + int(layer)
            scale_index[scale_val].append(idx)

        all_patches = []
        for idx, val in enumerate(scale_index):
            tmp_cv_kpts = [cv_kpts[i] for i in val]
            scale_img = self.pyr[idx]
            patches = self.get_interest_region(scale_img, tmp_cv_kpts, standardize=self.standardize)
            if patches is not None:
                all_patches.append(patches)

        if self.down_octave:
            all_patches = np.concatenate(all_patches[::-1], axis=0)
        else:
            all_patches = np.concatenate(all_patches, axis=0)
        assert len(cv_kpts) == all_patches.shape[0]
        return all_patches


class MatcherWrapper(object):
    """OpenCV matcher wrapper."""

    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def get_matches(self, feat1, feat2, cv_kpts1, cv_kpts2, ratio=None, cross_check=True,
                    err_th=4, ransac=True, info=''):
        """Compute putative and inlier matches.
        Args:
            feat1: (n_kpts, 128) Local features of src image.
            feat2: (n_kpts, 128) Local features of dst image.
            cv_kpts1: A list of keypoints represented as cv2.KeyPoint.
            cv_kpts2: A list of keypoints represented as cv2.KeyPoint.
            ratio: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            err_th: Epipolar error threshold.
            ransac: use ransac
            info: Info to print out.
        Returns:
            ok
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """

        init_matches1 = self.matcher.knnMatch(feat1, feat2, k=2)
        init_matches2 = self.matcher.knnMatch(feat2, feat1, k=2)

        good_matches = []
        for i in range(len(init_matches1)):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[init_matches1[i][0].trainIdx][0].trainIdx == i
                cond *= cond1
            if ratio is not None:
                cond2 = init_matches1[i][0].distance <= ratio * init_matches1[i][1].distance
                cond *= cond2
            if cond:
                good_matches.append(init_matches1[i][0])

        if not good_matches:
            return False, None, None

        if type(cv_kpts1) is list and type(cv_kpts2) is list:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx].pt for m in good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx].pt for m in good_matches])
        elif type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx] for m in good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx] for m in good_matches])
        else:
            return False, None, None

        if ransac:
            _, mask = cv2.findFundamentalMat(good_kpts1, good_kpts2, cv2.RANSAC, err_th, confidence=0.999)
            n_inlier = np.count_nonzero(mask)
            # print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)
        else:
            mask = np.ones((len(good_matches),))
            # print(info, 'n_putative', len(good_matches))
        return True, good_matches, mask

    @staticmethod
    def draw_matches(img1, cv_kpts1, img2, cv_kpts2, good_matches, mask,
                     match_color=(0, 255, 0), pt_color=(0, 0, 255)):
        """Draw matches."""
        if type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            cv_kpts1 = [cv2.KeyPoint(cv_kpts1[i][0], cv_kpts1[i][1], 1)
                        for i in range(cv_kpts1.shape[0])]
            cv_kpts2 = [cv2.KeyPoint(cv_kpts2[i][0], cv_kpts2[i][1], 1)
                        for i in range(cv_kpts2.shape[0])]
        display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches,
                                  None,
                                  matchColor=match_color,
                                  singlePointColor=pt_color,
                                  matchesMask=mask.ravel().tolist(), flags=4)
        return display


def sample_by_octave(cv_kpts, n_sample, down_octave=True):
    """Sample keypoints by octave.
    Args:
        cv_kpts: The list of keypoints representd as cv2.KeyPoint.
        n_sample: The sampling number of keypoint. Leave to -1 if no sampling needed
        down_octave: (True by default) Perform sampling downside of octave.
    Returns:
        npy_kpts: (n_kpts, 5) Keypoints in NumPy format, represenetd as
                  (x, y, size, orientation, octave).
        cv_kpts: A list of sampled cv2.KeyPoint.
    """

    n_kpts = len(cv_kpts)
    npy_kpts = np.zeros((n_kpts, 5))
    for idx, val in enumerate(cv_kpts):
        npy_kpts[idx, 0] = val.pt[0]
        npy_kpts[idx, 1] = val.pt[1]
        npy_kpts[idx, 2] = val.size
        npy_kpts[idx, 3] = val.angle * np.pi / 180.
        npy_kpts[idx, 4] = np.int8(val.octave & 0xFF)

    if down_octave:
        sort_idx = (-npy_kpts[:, 2]).argsort()
    else:
        sort_idx = (npy_kpts[:, 2]).argsort()

    npy_kpts = npy_kpts[sort_idx]
    cv_kpts = [cv_kpts[i] for i in sort_idx]

    if -1 < n_sample < n_kpts:
        # get the keypoint number in each octave.
        _, unique_counts = np.unique(npy_kpts[:, 4], return_counts=True)

        if down_octave:
            unique_counts = list(reversed(unique_counts))

        n_keep = 0
        for i in unique_counts:
            if n_keep < n_sample:
                n_keep += i
            else:
                break
        npy_kpts = npy_kpts[:n_keep]
        cv_kpts = cv_kpts[:n_keep]

    return npy_kpts, cv_kpts


def grayscale(image, c=3):
    """将输入图像转化为灰度图像

    :param image: numpy.array 输入图像
    :param c: int 输入图像通道数
    :return: grayscale_image
    """
    if c == 1:
        return image
    elif c == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif c == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        return None


def normalize_(image, mean, std):
    image -= mean
    image /= std


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_jittering_(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)


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

