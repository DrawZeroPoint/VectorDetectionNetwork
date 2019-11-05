#!/usr/bin/env python
# encoding: utf-8

# 可视化工具测试

import os
import cv2
import unittest


class Test(unittest.TestCase):

    def test_visualize_classification(self):
        """"""

        from utils.vis.util import visualize_classification
        pass

    def test_show_tensor_data(self):
        """为执行测试, 需要/Database文件夹下存在train/ val/两个文件夹, 每个文件夹下有若干
           子文件夹, 其中存有图像
        """
        from utils.io.util import get_dataloaders
        from utils.vis.util import show_tensor_data

        ok, dataloaders = get_dataloaders('/Database', 224, 1)
        if os.path.exists(os.path.join('/Database', 'train', 'a')) and \
                os.path.exists(os.path.join('/Database', 'val', 'a')):
            self.assertTrue(ok)
            self.assertIsNotNone(dataloaders)
            inputs, classes = next(iter(dataloaders['train'])) 
            show_tensor_data(inputs, title=['test' for x in classes])
        else:
            self.assertFalse(ok)
            self.assertIsNone(dataloaders)

    def test_print_functions(self):
        from utils.vis.util import print_info, print_warn, print_error

        print_info('This is an info')
        print_warn('This is a warning')
        print_error('THis is an error')

    def test_print_sys_info(self):
        from utils.vis.util import print_sys_info

        print_sys_info()

    def test_apply_frame(self):
        from utils.vis.util import apply_frame
        from PIL import Image, ImageDraw

        test_im = Image.open('/Database/Test/abnormal_detection/abnormal_detection.jpg')
        w, h = test_im.size

        draw = ImageDraw.Draw(test_im)
        # 测试正常情况下
        bbox1 = (30, 30, 180, 180)
        apply_frame(draw, bbox1, w, h, 1)
        # 测试框大于图像
        bbox2 = (30, 30, w + 1, h + 1)
        apply_frame(draw, bbox2, w, h)

        test_im.save('/Database/Test/abnormal_detection/abnormal_detection_bbox.jpg', 'jpeg')
        # 框的边长小与1的情况
        test_im = Image.open('/Database/Test/abnormal_detection/abnormal_detection.jpg')
        test_im.thumbnail((90, 90))
        bbox3 = (20, 20, 80, 80)
        apply_frame(draw, bbox3, w, h)
        test_im.save('/Database/Test/abnormal_detection/abnormal_detection9x9_bbox.jpg', 'jpeg')

        test_im2 = Image.open('/Database/Test/abnormal_detection/test2.jpg')
        draw = ImageDraw.Draw(test_im2)
        apply_frame(draw, bbox1, w, h)
        test_im2.save('/Database/Test/abnormal_detection/test1_1.jpg', 'jpeg')

    def test_apply_text(self):

        from utils.vis.util import apply_text
        from PIL import Image, ImageDraw

        test_im = Image.open('/Database/Test/abnormal_detection/abnormal_detection.jpg')
        w, h = test_im.size

        draw = ImageDraw.Draw(test_im)

        pt = [-20, -20]
        text = '测试文字[-10,-10]'
        apply_text(draw, pt, text, w, h)

        pt = [0, 0]
        text = '测试文字[0,0]'
        apply_text(draw, pt, text, w, h)

        pt = [60, 60]
        text = '测试文字[60,60]'
        apply_text(draw, pt, text, w, h)

        pt = [w, h]
        text = '测试文字[w,h]'
        apply_text(draw, pt, text, w, h)

        test_im.save('/Database/Test/abnormal_detection/abnormal_detection_text.jpg', 'jpeg')

    def test_apply_cross(self):
        from utils.vis.util import apply_cross
        from PIL import Image, ImageDraw

        test_im = Image.open('/Database/Test/abnormal_detection/abnormal_detection.jpg')
        w, h = test_im.size
        draw = ImageDraw.Draw(test_im)

        pt = [60, 60]
        apply_cross(draw, pt)

        pt = [0, 0]
        apply_cross(draw, pt)

        pt = [w+10, w-10]
        apply_cross(draw, pt)

        test_im.save('/Database/Test/abnormal_detection/abnormal_detection_cross.jpg', 'jpeg')

    def test_concat_image_list(self):
        from utils.vis.util import concat_image_list
        img1 = cv2.imread('/Database/Test/keypoints_matching/test1.jpg')
        img2 = cv2.imread('/Database/Test/keypoints_matching/test2.jpg')
        concat_image = concat_image_list([img1, img2])
        cv2.imwrite('/Database/Test/keypoints_matching/concat.jpg', concat_image)


if __name__ == '__main__':
    unittest.main()
