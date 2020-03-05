# coding: utf-8

import cv2
import copy
import math
import torch
import random
import logging
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from libs.utils.transforms import affine_transform
from libs.utils.transforms import get_affine_transform


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.pixel_std = 200
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT  # .jpg
        
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE  # w*h: 96 * 96
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, kp_preds, vd_preds, output_dir, all_boxes, img_path, *args, **kwargs):
        raise NotImplementedError

    def select_data(self, db):
        db_selected = []
        for record in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(record['joints_3d'], record['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = record['scale'][0] * record['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(record['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / (0.2**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(record)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def __len__(self,):
        return len(self.db)

    # @torchsnooper.snoop()
    def __getitem__(self, idx):
        db_record = copy.deepcopy(self.db[idx])

        image_file = db_record['image']
        filename = db_record['filename'] if 'filename' in db_record else ''
        imgnum = db_record['imgnum'] if 'imgnum' in db_record else ''

        if self.data_format == 'zip':
            from utils import zipreader
            input_numpy = zipreader.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            input_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        if input_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints_xyv = db_record['joints_xyv']

        c = db_record['center']
        s = db_record['scale']
        score = db_record['score'] if 'score' in db_record else 1
        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.5 else 0

        trans = get_affine_transform(c, s, r, self.image_size)
        input_t = cv2.warpAffine(input_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                 flags=cv2.INTER_LINEAR)

        if self.transform:
            input_t = Image.fromarray(input_t)
            input_t = self.transform(input_t)

        # joints_xyv [num_joints, k, 5] -> (x0, y0, x1, y1, v)
        # n is the keypoint number of each target, k is the instance number, here k==3
        # if in an image the instance number is less than 3, then for the absence instances v==0
        for n in range(self.num_joints):
            for k in range(joints_xyv.shape[1]):
                if joints_xyv[n][k][4] > 0:
                    joints_xyv[n, k, 0:2] = affine_transform(joints_xyv[n, k, 0:2], trans)
                    joints_xyv[n, k, 2:4] = affine_transform(joints_xyv[n, k, 2:4], trans)

        target_heatmap, target_vectormap = self.generate_target(joints_xyv)

        target_heatmap = torch.from_numpy(target_heatmap)
        target_vectormap = torch.from_numpy(target_vectormap)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'joints_xyv': joints_xyv
        }

        return input_t, target_heatmap, target_vectormap, meta

    def generate_target(self, joints_xyv: np.ndarray):
        """Get target heatmap and vectormap from joint labels.

        :param joints_xyv: [num_joints, k, 5], k is the object number; 5 is for [head_x, head_y, tail_x, tail_y, vis]
        :return: target_heatmap: [num_joints, h, w]
                 target_vectormap: [num_joints, 2, h, w]; 2 for (vx, vy) ∈ [-1, 1]
        """
        assert self.target_type == 'gaussian', 'Only support gaussian map now!'

        target_heatmap = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        target_vectormap = np.zeros((self.num_joints, 2, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

        tmp_size = self.sigma * 3  # 3*3

        for n in range(self.num_joints):
            for k in range(joints_xyv.shape[1]):
                if joints_xyv[n][k][-1] == 0:
                    continue

                feat_stride = self.image_size / self.heatmap_size
                head_x = int(joints_xyv[n][k][0] / feat_stride[0] + 0.5)
                head_y = int(joints_xyv[n][k][1] / feat_stride[1] + 0.5)

                # for generating the vector map, we calculate the end point (tail_x, tail_y) of each vector
                tail_x = int(joints_xyv[n][k][2] / feat_stride[0] + 0.5)
                tail_y = int(joints_xyv[n][k][3] / feat_stride[1] + 0.5)

                # Check that any part of the gaussian is in-bounds
                ul = [int(head_x - tmp_size), int(head_y - tmp_size)]
                br = [int(head_x + tmp_size + 1), int(head_y + tmp_size + 1)]

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]

                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                # determine the heat value of the pixel affected by multiple peaks to be the maximum value
                prev_val = target_heatmap[n][img_y[0]:img_y[1], img_x[0]:img_x[1]]
                curr_val = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                max_val = np.maximum(prev_val, curr_val)
                target_heatmap[n][img_y[0]:img_y[1], img_x[0]:img_x[1]] = max_val

                dx = head_x - tail_x
                dy = head_y - tail_y

                dl = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
                # unit vector (vx, vy)
                vx = dx / dl
                vy = dy / dl

                # If due to joints adjacent, two or more vectormap overlapped, calculate the mean value
                prev_vx = target_vectormap[n][0][img_y[0]:img_y[1], img_x[0]:img_x[1]]
                target_vectormap[n][0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.where(prev_vx == 0, vx,
                                                                                        (prev_vx + vx) * 0.5)

                prev_vy = target_vectormap[n][1][img_y[0]:img_y[1], img_x[0]:img_x[1]]
                target_vectormap[n][1][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.where(prev_vy == 0, vy,
                                                                                        (prev_vy + vy) * 0.5)

        return target_heatmap, target_vectormap
