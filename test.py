#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import cv2
import os
import unittest
import numpy as np

import utils.vis.util as vis_util
import libs.core.inference as lib_inference


class Test(unittest.TestCase):

    def test_get_all_preds(self):
        batch_heatmaps = np.load("/VDN/data/test.npy")
        vis_util.save_np_heatmaps(batch_heatmaps, "/VDN/data/test_heatmap_max.jpg", is_max=True)
        vis_util.save_np_heatmaps(batch_heatmaps, "/VDN/data/test_heatmap_all.jpg", is_max=False)
        # lib_inference.get_all_preds(batch_heatmaps)

    # def test_port_port(self):
    #     """"""
    #     self.main_dir = "/Database/Test/0_meter_recognition/classficition"
    #     self.template_json_dir = "/Database/Test/0_meter_recognition/Wuhan/out_door_v0.1/json"
    #     self.template_jpg_dir = "/Database/Test/0_meter_recognition/Wuhan/out_door_v0.1/image"
    #     file_list = os.listdir(self.main_dir)
    #     port = Port()
    #     for item in file_list:
    #         image = self.main_dir + '/' + item
    #         print(image)
    #         src_img = cv2.imread(image)
    #         template_image = self.template_jpg_dir + '/' + item[:-7] + '.png'
    #         template_info = self.template_json_dir + '/' + item[:-7] + '.json'
    #         print(template_info)
    #         logger.info(f"src{image}")
    #         logger.info(f"template{template_image}")
    #         err, output, marked_image = port.process(src_img, template_info, template_image)
    #         print(item)
    #         print('err:', err)
    #         print('output:', output)
    #         imagesave_path = '/Database/Test/0_meter_recognition/classficition_result/out-' + item
    #         cv2.imwrite(imagesave_path, marked_image)
    # def test_port_port(self):
    #     """"""
    #     self.main_dir = "/Database/Test/0_meter_recognition/0_test/20191030/20191030task"
    #     self.template_json_dir = "/Database/Test/0_meter_recognition/0_test/20191030/20191030deploy"
    #     self.template_jpg_dir = "/Database/Test/0_meter_recognition/0_test/20191030/20191030deploy"
    #     file_list = os.listdir(self.main_dir)
    #     port = Port()
    #     for item in file_list:
    #         image = self.main_dir + '/' + item
    #         print(image)
    #         src_img = cv2.imread(image)
    #         template_image = self.template_jpg_dir + '/' + item[:-3] + 'jpg'
    #         template_info = self.template_json_dir + '/' + item[:-3] + 'json'
    #         print(template_info)
    #         logger.info(f"src{image}")
    #         logger.info(f"template{template_image}")
    #         err, output, marked_image = port.process(src_img, template_info, template_image)
    #         # err, output, marked_image = port.process(src_img)
    #         print(item)
    #         print('err:', err)
    #         print('output:', output)
    #         imagesave_path = '/Database/Test/0_meter_recognition/0_test/20191030/20191030task_out/out-' + item
    #         cv2.imwrite(imagesave_path, marked_image)

    # def test_port_port(self):
    #     """"""
    #     self.main_dir = "/Database/Raw/Visual/Image/pengcheng190920/meter1"
    #     # self.template_json_dir = "/Database/Test/0_meter_recognition/no_rec_json"
    #     # self.template_jpg_dir = "/Database/Test/0_meter_recognition/no_rec_png"
    #     file_list = os.listdir(self.main_dir)
    #     port = Port()
    #     for item in file_list:
    #         image = self.main_dir + '/' + item
    #         print(image)
    #         src_img = cv2.imread(image)
    #         # template_image = self.template_jpg_dir + '/' + item[:-3] + 'png'
    #         # template_info = self.template_json_dir + '/' + item[:-3] + 'json'
    #         # print(template_info)
    #         # logger.info(f"src{image}")
    #         # logger.info(f"template{template_image}")
    #         err, output, marked_image = port.process(src_img)
    #         print(item)
    #         print('err:', err)
    #         print('output:', output)
    #         imagesave_path = '/Database/Raw/Visual/Image/pengcheng190920/meter1_test_rsult/out-' + item
    #         cv2.imwrite(imagesave_path, marked_image)


if __name__ == '__main__':
    unittest.main()
