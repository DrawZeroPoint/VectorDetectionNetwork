# coding: utf-8

import cv2
import copy
import math
import torch
import random
import logging
import torchsnooper
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from libs.utils.transforms import affine_transform
from libs.utils.transforms import get_affine_transform


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 1
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT  # .jpg
        
        self.flip = cfg.DATASET.FLIP
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

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / (0.2**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def __len__(self,):
        return len(self.db)

    # @torchsnooper.snoop()
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            input_numpy = zipreader.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            input_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        if input_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints_xyv = db_rec['joints_xyv']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
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
        for i in range(self.num_joints):
            for k in range(joints_xyv.shape[1]):
                if joints_xyv[i][k][4] > 0:
                    joints_xyv[i, k, 0:2] = affine_transform(joints_xyv[i, k, 0:2], trans)
                    joints_xyv[i, k, 2:4] = affine_transform(joints_xyv[i, k, 2:4], trans)

        target_heatmap, target_vector, tgt_indexes = self.generate_target(joints_xyv)

        target_heatmap = torch.from_numpy(target_heatmap)
        target_vector = torch.from_numpy(target_vector)
        tgt_indexes = torch.from_numpy(tgt_indexes)

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

        return input_t, target_heatmap, target_vector, tgt_indexes, meta

    # @torchsnooper.snoop()
    def generate_target(self, joints_xyv):
        """k is the number of keypoints in each heatmap

        :param joints_xyv:  [num_joints, k, 5], k is the object number; 5 -> [x0, y0, x1, y1, vis]
        :return: target_heatmap [num_joints, h, w]
                 target_vector  [num_joints, k, 2]  (vx, vy) -> [-1, 1]
                 target_indexes [num_joints, k]  idx are the peaks on heatmap
        """
        target_vector = np.zeros((self.num_joints, joints_xyv.shape[1], 2), dtype=np.float32)
        target_indexes = np.zeros((self.num_joints, joints_xyv.shape[1]), dtype=np.long)

        assert self.target_type == 'gaussian', 'Only support gaussian map now!'

        target_heatmap = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        tmp_size = self.sigma * 3  # 3*3

        for j in range(self.num_joints):
            for k in range(joints_xyv.shape[1]):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints_xyv[j][k][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_xyv[j][k][1] / feat_stride[1] + 0.5)

                # for generating vectors
                x1 = int(joints_xyv[j][k][2] / feat_stride[0] + 0.5)
                y1 = int(joints_xyv[j][k][3] / feat_stride[1] + 0.5)

                target_indexes[j][k] = mu_x + mu_y * self.heatmap_size[1]

                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

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

                prev_val = target_heatmap[j][img_y[0]:img_y[1], img_x[0]:img_x[1]]
                curr_val = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                combined_val = np.maximum(prev_val, curr_val)
                # print(f'com {combined_val}')
                target_heatmap[j][img_y[0]:img_y[1], img_x[0]:img_x[1]] = combined_val

                dx = x1 - mu_y
                dy = y1 - mu_y

                if dy != 0 and dx != 0:
                    vx = math.sqrt(1 / (1 + math.pow(dx, 2) / math.pow(dy, 2))) * (dx / math.fabs(dx))
                    vy = math.sqrt(1 - math.pow(vx, 2)) * (dy / math.fabs(dy))
                else:
                    if dy == 0:
                        vx = 1
                        vy = 0
                    else:
                        vy = 1
                        vx = 0

                target_vector[j, k, :] = np.array([vx, vy])

        return target_heatmap, target_vector, target_indexes
