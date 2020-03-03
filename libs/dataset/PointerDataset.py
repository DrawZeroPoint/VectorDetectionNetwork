#!/usr/bin/env python
# encoding: utf-8

import logging
import os
import copy
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
        self.coco = COCO(self._get_ann_file())
        self.coco_vds = self.generate_vds(copy.deepcopy(self.coco))

        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.max_joint_num = 3  # TODO check this
        self.max_instance_num = 3
        self.parent_ids = None

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file(self):
        """self.image_set could be train_pointer, val_pointer, or test_pointer.
        Note that be different with COCO, we provide annotations for the test split,
        so the 'image_info' prefix is not applied to test_pointer

        """
        filename = os.path.join(self.root, 'annotations', 'ann_' + self.image_set + '.json')
        print(f'The annotation file: {filename}')
        return filename

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        return self._load_coco_keypoint_annotations()

    def _load_coco_keypoint_annotations(self):
        """ground truth bbox and keypoints.

        """
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

        # one img id is correspond to multiple annotation ids, each annotation is for a pointer,
        # but these pointers all share the same bounding box.
        ann_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        shared_bbox = None
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                shared_bbox = [x1, y1, x2 - x1, y2 - y1]
                break

        if shared_bbox is None:
            raise ValueError('No shared bbox available')

        center, scale = self._box2cs(shared_bbox)

        vectors = []
        for k, obj in enumerate(objs):
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                # Here we use 1 to represent the only class (meter) of the Pointer-10K dataset
                raise ValueError(f'class name {cls} is not 1, please check the label')

            # Ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                raise ValueError('No labeled keypoint in given object.')

            # We use the head of the pointer as the target while using the tail to calculate the vector
            x0 = obj['keypoints'][0]  # (x0, y0) is the tip
            y0 = obj['keypoints'][1]
            x1 = obj['keypoints'][6]  # (x1, y1) is the tail
            y1 = obj['keypoints'][7]
            vis = obj['keypoints'][2]

            pt = np.array([x0, y0, x1, y1, vis])
            vectors.append(pt)

        # This process make the number of pointers within each bbox equals to 3
        sz = len(vectors)
        if sz < self.max_instance_num:
            i = self.max_instance_num - sz
            while i:
                vectors.append(np.array([0, 0, 0, 0, 0]))
                i -= 1

        vectors_array = np.array(vectors)
        res = np.expand_dims(vectors_array, 0)

        record = [{
            'image': self.image_path_from_index(index),
            'center': center,
            'scale': scale,
            'joints_xyv': res,  # joints_xyv [num_joints, k==3, 5]
            'filename': '',
            'imgnum': 0,
        }]

        return record

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
        """example: images / train_pointer / 000000119993.jpg.

        """
        file_name = '%012d.jpg' % index
        prefix = self.image_set
        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(self.root, 'images', data_name, file_name)

        return image_path

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, *args, **kwargs):
        """

        :param cfg:
        :param preds: (num_samples, max_instance_num, 3)
        :param output_dir:
        :param all_boxes: (num_samples, 6)
        :param img_path:
        :param args:
        :param kwargs:
        :return:
        """
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, 'keypoints_%s_results.json' % self.image_set)

        preds_list = []
        for idx, kpt in enumerate(preds):
            preds_list.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })

        kpts = defaultdict(list)
        for kpt in preds_list:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = 3
        in_vis_thre = self.in_vis_thre  # 0.2
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
                        valid_num += 1

                if valid_num != 0:
                    kpt_score = kpt_score / valid_num

                n_p['score'] = kpt_score * box_score

            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)

        info_str = self._do_python_keypoint_eval(res_file, res_folder)
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']

    def evaluate_vds(self, cfg, preds, output_dir, all_boxes, img_path, *args, **kwargs):
        """

        :param cfg:
        :param preds: (num_samples, max_instance_num, 3)
        :param output_dir:
        :param all_boxes: (num_samples, 6)
        :param img_path:
        :param args:
        :param kwargs:
        :return:
        """
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, 'vds_%s_results.json' % self.image_set)

        preds_list = []
        for idx, kpt in enumerate(preds):
            preds_list.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })

        kpts = defaultdict(list)
        for kpt in preds_list:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = 3
        in_vis_thre = self.in_vis_thre  # 0.2
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
                        valid_num += 1

                if valid_num != 0:
                    kpt_score = kpt_score / valid_num

                n_p['score'] = kpt_score * box_score

            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)

        info_str = self._do_python_vds_eval(res_file, res_folder)
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']

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
        except FileNotFoundError:
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

            _key_points = np.array([img_kpts[k]['keypoints'] for k in range(len(img_kpts))])
            key_points = np.zeros((_key_points.shape[0], self.max_joint_num * 3), dtype=np.float)

            for ipt in range(self.max_joint_num):
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
        coco_dt = self.coco.load_res(res_file)
        # print('PointerDataset 329 coco dt', coco_dt.anns, coco_dt.cats)
        # print('PointerDataset 330 coco gt', self.coco.anns, self.coco.cats)
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

    def _do_python_vds_eval(self, res_file, res_folder):
        coco_dt = self.coco.load_res(res_file)
        # print('PointerDataset 329 coco dt', coco_dt.anns, coco_dt.cats)
        # print('PointerDataset 330 coco gt', self.coco.anns, self.coco.cats)
        coco_eval = COCOeval(self.coco_vds, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        eval_file = os.path.join(res_folder, 'vds_%s_results.pkl' % self.image_set)

        with open(eval_file, 'wb') as f:
            pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
        logger.info('=> VDS eval results saved to %s' % eval_file)

        return info_str

    def generate_vds(self, coco_cp):
        anns = coco_cp.anns

        assert 'keypoints' in anns[0]
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            vx = x[0] - x[-1]
            vy = y[0] - y[-1]
            rad = np.arctan2(vy, vx)
            deg = np.rad2deg(rad)

            x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x1 - x0) * (y1 - y0)
            ann['id'] = id + 1
            ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]

            ann['keypoints'][0:2] = deg

        return coco_cp
