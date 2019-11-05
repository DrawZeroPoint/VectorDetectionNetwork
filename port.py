#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 表计示数识别模组
import numpy as np
import cv2
from PIL import ImageDraw, Image

from modules.meter_recognition_v1.tools import kx_detect, kx_tools, kx_pointer_pose, kx_digit_classification, \
    kx_get_readings_intersection

import utils.io.util as io_util
import utils.cv.util as cv_util
import utils.vis.util as vis_util
# import modules.meter_recognition_v1.tools.kx_match_tfeat as kpm_port
import modules.keypoints_matching_v1.port as kpm_port
from modules.meter_recognition_v1.tools.kx_get_readings_intersection import least_squares
import heapq
from utils.log.log_cfg import Log

logger = Log(__name__).get_log()

KEYPOINTS_NUM = 5


class Object(object):
    def __init__(self):
        self.id = None
        self.reading = None
        self.trust = 0.0
        self.precision = 0


def load_pointer_pose_net():
    return kx_pointer_pose.PointerPoseNet()


def load_digit_net():
    return kx_digit_classification.DigitNet()


def check_part_info_list(part_info):
    """ 检查part_info某个属性不为空

    :param part_info:
    :return:
    """
    if 'bbox' in part_info and 'scale_points' in part_info and 'scale_numbers' in part_info and 'id' in part_info:
        if not part_info['bbox'] or not part_info['scale_points'] or not part_info['scale_numbers']:
            return False
        else:
            return True
    else:
        return False


def get_pointer_pose(points_x, points_y, k, b, img, score):
    """获取指针线线段：给定预测点横坐标、拟合直线斜率和截距以及ROI，获取线段端点坐标

    :param score: 预测点置信度
    :param k: 拟合直线斜率
    :param b: 拟合直线截距
    :param points_y: 预测点纵坐标
    :param points_x: 预测点横坐标
    :param img: ROI(即整个仪表)
    :return: p1_x, p1_y, p2_x, p2_y
    """
    p1_x = 0
    p1_y = 0
    p2_x = 0
    p2_y = 0
    reading_num = len(points_x) / 5
    for j in range(int(reading_num)):
        start = KEYPOINTS_NUM * j
        score_list = []
        for i in range(KEYPOINTS_NUM):
            score_list.append(float(score[i][0]))
        index_list = list(map(score_list[:-1].index, heapq.nlargest(2, score_list[:-1])))
        x_start = points_x[min(index_list)]
        y_start = points_y[min(index_list)]
        x_end = points_x[max(index_list)]
        y_end = points_y[max(index_list)]
        p1_x = int(x_start)
        if k is None:
            p1_y = points_y[start]
            if y_end <= y_start:
                p2_y = 0
            else:
                p2_y = img.shape[0]  # shape.h
            p2_x = p1_x
        else:
            if x_end < x_start:
                p1_y = k * p1_x + b
                p2_x = 0
                p2_y = k * p2_x + b
            elif x_end > x_start:
                p1_y = k * p1_x + b
                p2_x = img.shape[1]  # shape.w
                p2_y = k * p2_x + b
            else:
                p1_y = k * p1_x + b
                p2_x = p1_x
                p2_y = 0
    return p1_x, p1_y, p2_x, p2_y


class Port:
    def __init__(self):
        self.detect_net = kx_detect.DetectNet(True, train=False)
        self.kpm_api = kpm_port.Port()
        self.pointer_pose_net = load_pointer_pose_net()
        self.digit_net = load_digit_net()
        self.dataset_io = kx_tools.DatasetIO()
        self.recognition_inter = kx_get_readings_intersection.XMethod()

    @staticmethod
    def _draw_marked_image(cv_image, kp_xy, score):
        """将识别结果绘制在原图像上, 输出OpenCV格式有指针线标记的图像

        :param cv_image: numpy.array 输入OpenCV格式图像
        :param kp_xy: list -> [x, y] 预测关键点坐标列表
        :param score: ndarray 关键点置信度数组
        :return: numpy.array marked_image
        """
        ok, w, h, c = cv_util.check_cv_image_status(cv_image)
        if score.mean() > 0.1:
            pil_img = Image.fromarray(cv_image)
            draw = ImageDraw.Draw(pil_img)

            points_x = []
            points_y = []
            for i, mat in enumerate(kp_xy):
                x, y = int(mat[0]), int(mat[1])
                points_x.append(x)
                points_y.append(y)
                if score[i] > 0.1:
                    vis_util.apply_dot(draw, [x, y], w * 0.1, h * 0.1, i)

            np_x = np.array(points_x)
            np_y = np.array(points_y)
            flag, coef, intercept_ = least_squares(np_x, np_y)
            line = get_pointer_pose(points_x, points_y, coef, intercept_, cv_image, score)
            vis_util.apply_line(draw, line, w * 0.5, h * 0.5, 0)
            image = np.array(pil_img)
        else:
            image = cv_image
        return image

    @staticmethod
    def _draw_result(cv_image, item):
        """将识别结果绘制在原图像上

        :param cv_image: numpy.array 输入OpenCV格式图像
        :param item: 识别结果字典
        :return marked_image: numpy.array 输出OpenCV格式有指针线及读数标记的图像
        """
        img = Image.fromarray(cv_image)
        draw = ImageDraw.Draw(img)
        class_name = str(round(item.reading, item.precision))
        vis_util.apply_text(draw, [int(img.height * 0.05), int(img.width * 0.05)], class_name,
                            img.width, img.height, 19, bg=True)
        return np.array(img)

    def _rectify(self, src_img, tmp_img, state=True):
        """反馈由src图像到tmp图像的单应性变换矩阵

        :param src_img: numpy.array -> BGR 输入源图像
        :param tmp_img: numpy.array -> BGR 输入模板图像
        :return: err, h
        """
        ok, h, temp_name, good_num = self.kpm_api.process(src_img, tmp_img, state)
        if not ok:
            return 30003, None, None, None
        else:
            return 0, h, temp_name, good_num

    @staticmethod
    def _get_most_overlap_part(bbox_temp, bboxes):
        """根据模板图像上经H矩阵变换得到的bbox, 在bboxes中搜索与之最重合ROI的id并返回

        :param bbox_temp: list -> int [x_lt, y_lt, x_rb, y_rb]
        :param bboxes: list -> int[[x_lt, y_lt, x_rb, y_rb]...]
        :return: ok, roi_id
        """
        max_iou = 0
        roi_id = -1
        for i in range(len(bboxes)):
            bbox_dst = bboxes[i]
            iou = cv_util.get_bbox_iou(bbox_temp, bbox_dst)
            if iou >= max_iou:
                max_iou = iou
                roi_id = i
        if roi_id == -1:
            return 10006, None
        else:
            return 0, roi_id

    def _get_results(self, kp_xy_temp, part_info, score):
        """通过检测得到源图像上包含指针式仪表的ROI, 前序获得该ROI的bbox和其中代表指针的5个关键点kp_xy,
         再根据该part info获得表计读值及其置信度, 作为item返回

        :param kp_xy_temp: list -> [x, y] 预测的指针线关键点转换到模板坐标系
        :param part_info: dict 部件信息列表
        :return: ok, item
        """

        meter_id = part_info['id']
        try:
            precision = part_info["precision"]
        except KeyError:
            precision = 3
        if not precision:
            precision = 3

        err_reading, reading, trust = self.recognition_inter.read_with_x_method(kp_xy_temp, part_info, score)

        if err_reading == 0:
            logger.info("读值是：{}".format(reading))
            item = Object()
            item.id = meter_id
            item.reading = reading[0]  # TODO 去除list
            item.trust = trust[0]
            item.precision = precision
            return 0, item
        else:
            logger.error("未获取读值")
            return 10008, None

    # TODO 简化参数量
    def verbose(self, crop_temp_image, kp_xy, kp_xy_trans, score, meter_id, part_info, temp_image, kp_xy_tmp, roi,
                bbox):
        """保存算法中间过程图像便于调试
        :param kp_xy_trans: list -> [x, y] 预测的指针线关键点转换到模板坐标系
        :param part_info: dict 部件信息列表
        :return:
        """
        marked_image_1 = self._draw_marked_image(crop_temp_image, kp_xy_trans, score, )
        cv2.imwrite('/Database/Test/0_meter_recognition/0_test/cache/{}_image.jpg'.format(meter_id), marked_image_1)
        marked_image_3 = self._draw_marked_image(roi, kp_xy, score)
        cv2.imwrite('/Database/Test/0_meter_recognition/0_test/cache/{}_image_roi.jpg'.format(meter_id), marked_image_3)
        marked_image_2 = self._draw_marked_image(temp_image, kp_xy_tmp, score)
        pil_img = Image.fromarray(marked_image_2)
        draw = ImageDraw.Draw(pil_img)

        scale_pt_list = part_info['scale_points']
        points_x = []
        points_y = []
        vis_util.apply_dot(draw, [bbox[0], bbox[1]], roi.shape[0], roi.shape[1], 2)
        for i in range(int(len(scale_pt_list) / 2)):
            x = scale_pt_list[2 * i]
            y = scale_pt_list[2 * i + 1]
            points_x.append(x)
            points_y.append(y)
            # if score[i] > 0.1:
            vis_util.apply_dot(draw, [x, y], roi.shape[0], roi.shape[1], i)
        image_1 = np.array(pil_img)
        cv2.imwrite('/Database/Test/0_meter_recognition/0_test/cache/{}_marked_image.jpg'.format(meter_id), image_1)

    # TODO 简化参数量
    def get_response(self, part_info_list, temp_image, src_image, bboxes, rois, h, kp_xy_list, score_list,
                     marked_image_list1):
        """遍历模板信息中需要识别的bbox，在检测出的bboxes中做overlap获取相应roi，
        然后进行局部图像配准（去除背景干扰信息），获取表计示数
        :param part_info_list: list 模板注释信息列表
        :param temp_image: numpy_array 模板图像
        :param src_image: numpy_array 待识别原图
        :param bboxes: list 目标检测出的表盘bboxes
        :param rois: numpy_array 表盘图像
        :param h: 全景图单应性矩阵
        :param kp_xy_list: list 所有的检测到的指针关键点组成的列表
        :param score_list: numpy_array 指针关键点置信度列表
        :param marked_image_list1: numpy_array 绘有指针线的图像
        :return: output, marked_image_list2
        """
        output = []
        marked_image_list2 = []
        logger.info('待识别表记数量{}'.format(len(part_info_list)))
        for part_info in part_info_list:
            err = check_part_info_list(part_info)
            if not err:
                logger.info('模板标注信息内容有缺')
            else:
                bbox = part_info['bbox']
                meter_id = part_info['id']
                logger.info('表{}识别'.format(meter_id))
                bbox_tmp = cv_util.transform_by_h([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], h, dim=1)
                bbox_tmp = list(map(int, bbox_tmp))
                err_overlap, roi_id = self._get_most_overlap_part(bbox_tmp, bboxes)
                logger.info('roi_id:{}'.format(roi_id))
                if err_overlap == 0:
                    crop_temp_image = temp_image[int(bbox[1]):int(bbox[3]),
                                      int(bbox[0]):int(bbox[2])]

                    if not crop_temp_image.any():
                        logger.error("模板图像和标注文档不匹配")
                    else:
                        # 首先roi局部配准，匹配点>20则通过，否则源图与模板全景图片配准
                        err_rectify_local_1, h_roi_1, temp_name_1, good_num_1 = \
                            self._rectify(rois[roi_id], crop_temp_image)
                        if h_roi_1 is not None:
                            err_rectify_local, h_roi = err_rectify_local_1, h_roi_1
                        else:
                            err_rectify_local_2, h_roi_2, temp_name_2, good_num_2 = self._rectify(src_image, temp_image)
                            err_rectify_local, h_roi = err_rectify_local_2, h_roi_2
                        if err_rectify_local == 0:
                            kp_xy_trans = cv_util.transform_by_h(kp_xy_list[roi_id], h_roi, dim=2)
                            kp_xy_tmp = np.add(np.array(kp_xy_trans), np.array([[bbox[0], bbox[1]]])).tolist()
                            self.verbose(crop_temp_image, kp_xy_list[roi_id], kp_xy_trans, score_list[roi_id],
                                         part_info['id'], part_info, temp_image, kp_xy_tmp, rois[roi_id], bbox)
                            err_result, item = self._get_results(kp_xy_tmp, part_info, score_list[roi_id])
                            if err_result == 0:
                                output.append(item)
                                marked_image = self._draw_result(marked_image_list1[roi_id], item)
                                marked_image_list2.append(marked_image)
                            else:
                                item = Object()
                                item.id = meter_id
                                item.reading = 10000
                                output.append(item)
                                logger.info("读值是：{}".format(item.reading))
                                logger.error('指针超量程')
                                marked_image_list2.append(marked_image_list1[roi_id])
                        else:
                            logger.error("局部图获取单应性矩阵失败")
                else:
                    logger.error("未在检测出的bboxes中获取与之最重合的bbox")
        return output, marked_image_list2

    def process(self, src_image, part_info_list=None, temp_image=None):
        """表计识别, 输出错误码, 结果字典, 标记图像

        :param src_image: numpy.ndarray -> BGR 输入表计图像
        :param part_info_list: list -> dict 或 str: 部件标注信息或其地址
        :param temp_image: numpy.ndarray -> BGR 或 str: 输入模板图像或其地址
        :return: err, output_dict, marked_image
        """

        if src_image is None:
            logger.error("输入数据错误")
            return 11009, None, None

        # 表计类目标的内部统一标识号为10002
        got, bboxes, rois, how_good, types_out = self.detect_net.detect(src_image, ['10002'], 0.8)
        if not got:
            logger.error("检测表盘失败")
            return 11009, None, src_image

        # 检测所有ROI指针线，并绘制检测指针线结果
        kp_xy_list = []
        score_list = []
        marked_image_list1 = []
        for i in range(got):
            roi_img = rois[i]
            kp_xy, score = self.pointer_pose_net.get_pointer_keypoints(roi_img)
            kp_xy_list.append(kp_xy)
            score_list.append(score)
            marked_image = self._draw_marked_image(roi_img, kp_xy, score)
            marked_image_list1.append(marked_image)

        # 模板为空时，在服务器模板库进行搜索，否则进行模板图与待测全景图（利于id识别及添加静态标志物）匹配，获取单应性矩阵
        if temp_image is not None:
            _, part_info_list = io_util.part_info_read(part_info_list)
            _, temp_image = io_util.imread(temp_image)
            err_rectify, h, _, _ = self._rectify(temp_image, src_image, state=False)
            # 配准失败，增强亮度，进行第二次配准（默认模板为清晰良好光照条件下采集的图像）
            # TODo 进行测试，封装为函数
            if h is None:
                src_image = np.uint8(np.clip((1.5 * src_image + 10), 0, 255))
                err_rectify, h, _, _ = self._rectify(temp_image, src_image, state=False)
        else:
            err_rectify, h, temp_name, _ = self._rectify(temp_image, src_image, state=False)
            part_info_list, temp_image = io_util.get_template_info(temp_name)
            _, part_info_list = io_util.part_info_read(part_info_list)
        output = []
        marked_image_list2 = []
        if err_rectify == 0:
            if temp_image is None or part_info_list is None:
                logger.warning('模板输入信息失败')
            else:
                output, marked_image_list2 = self.get_response(part_info_list, temp_image, src_image, bboxes, rois,
                                                               h, kp_xy_list, score_list, marked_image_list1)
        elif err_rectify == 30003:
            logger.warning('全景图获取单应性矩阵失败')
            if temp_image is None or part_info_list is None:
                logger.warning('模板库无对应模板')
            # 在全景校正失败的情况下, 假定源图像与模板图像均在同一位置由同一设备采集, 则目标在两张图像上的
            # 位置应较为接近, 此时直接将源图像上目标位置作为模板上目标的位置, 仍有可能对其正确读值
            else:
                output, marked_image_list2 = self.get_response(part_info_list, temp_image, src_image, bboxes, rois,
                                                               h, kp_xy_list, score_list, marked_image_list1)
        if not output:
            logger.warning('无示数输出')
            return 11009, None, vis_util.concat_image_list(marked_image_list1)
        else:
            output_dict = [ob.__dict__ for ob in output]
            return 0, output_dict, vis_util.concat_image_list(marked_image_list2)
