#!/usr/bin/env python
# encoding: utf-8

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np

# Notice that here we use customized COCO tools
from compiled.pycocotools.pycocotools.coco import COCO
from compiled.pycocotools.pycocotools.cocoeval import COCOeval

from libs.dataset.JointsDataset import JointsDataset
from compiled.keypoints.nms.nms import oks_nms


logger = logging.getLogger(__name__)


class PointerDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super(PointerDataset, self).__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.parent_ids = None

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """self.image_set could be train_pointer or val_pointer

        """
        prefix = 'ann' if 'test' not in self.image_set else 'image_info'
        filename = os.path.join(self.root, 'annotations', prefix + '_' + self.image_set + '.json')
        print(f'! ann file keypoints: {filename}')
        return filename

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        ann_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        all_in_one_obj = None
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                all_in_one_obj = obj

        if all_in_one_obj is None:
            raise ValueError('No bbox available')

        rec = []
        center, scale = self._box2cs(all_in_one_obj['clean_bbox'][:4])

        for j in range(self.num_joints):
            keys_of_joint_j = []
            for k, obj in enumerate(objs):
                cls = self._coco_ind_to_class_ind[obj['category_id']]
                if cls != 1:
                    raise ValueError('class not equal to 1')

                # Ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    raise ValueError('no keypoint')

                x = obj['keypoints'][j * 3]
                y = obj['keypoints'][j * 3 + 1]
                vis = obj['keypoints'][j * 3 + 2]

                pt = np.array([x, y, vis])
                keys_of_joint_j.append(pt)

            if j == 0:
                p1s = np.array(keys_of_joint_j)
            elif j == 1:
                p2s = np.array(keys_of_joint_j)
            else:
                p3s = np.array(keys_of_joint_j)

        res = np.concatenate((np.expand_dims(p1s, 0), np.expand_dims(p2s, 0), np.expand_dims(p3s, 0)))

        # joints [batch, num_joints, k, 3]
        # joint_vis [batch, num_joints, k, 3]

        rec.append({
            'image': self.image_path_from_index(index),
            'center': center,
            'scale': scale,
            'joints_xyv': res,
            'filename': '',
            'imgnum': 0,
        })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(self.root, 'images', data_name, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after filter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, *args, **kwargs):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, 'keypoints_%s_results.json' % self.image_set)

        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })

        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []

        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):


                    trust_score = n_p['keypoints'][n_jt][2]
                    if trust_score > in_vis_thre:
                        kpt_score = kpt_score + trust_score
                        valid_num = valid_num + 1

                if valid_num != 0:
                    kpt_score = kpt_score / valid_num

                # rescoring
                n_p['score'] = kpt_score * box_score
            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)

        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                      'cls_ind': cls_ind,
                      'cls': cls,
                      'ann_type': 'keypoints',
                      'keypoints': keypoints
                      }
                     for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float)

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [{'image_id': img_kpts[k]['image'],
                       'category_id': cat_id,
                       'keypoints': list(key_points[k]),
                       'score': img_kpts[k]['score'],
                       'center': list(img_kpts[k]['center']),
                       'scale': list(img_kpts[k]['scale'])
                       } for k in range(len(img_kpts))]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        eval_file = os.path.join(res_folder, 'keypoints_%s_results.pkl' % self.image_set)

        with open(eval_file, 'wb') as f:
            pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
        logger.info('=> coco eval results saved to %s' % eval_file)

        return info_str
