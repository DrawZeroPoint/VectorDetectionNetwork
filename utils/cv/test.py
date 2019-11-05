#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 图像处理工具测试

import os
import cv2
import unittest
import torchsnooper

from utils.cv.util import *


class Test(unittest.TestCase):

    @torchsnooper.snoop()
    def test_Siftwrapper(self):
        wrapper = SiftWrapper()
        wrapper.create()
        image = cv2.imread('/Database/Test/abnormal_detection/test.jpg')
        gray_img = grayscale(image)
        npy_kpts, cv_kpts = wrapper.detect(gray_img)
        sift_desc = wrapper.compute(image, cv_kpts)
        wrapper.build_pyramid(gray_img)
        all_patches = wrapper.get_patches(cv_kpts)

    @torchsnooper.snoop()
    def test_MatcherWrapper(self):
        matcher_wrapper = MatcherWrapper()

    @torchsnooper.snoop()
    def test_sample_by_octave(self):
        # sample_by_octave(cv_kpts, n_sample, down_octave=True)
        n_sample = 8192
        nfeatures = 0
        n_octave_layers = 3
        peak_thld = 0.0067
        edge_thld = 10
        sigma = 1.6
        sift = cv2.xfeatures2d.SIFT_create(nfeatures, n_octave_layers,
                                           peak_thld, edge_thld, sigma)

        image = cv2.imread('/Database/Test/abnormal_detection/test.jpg')
        gray_img = grayscale(image)
        cv_kpts = sift.detect(gray_img, None)
        npy_kpts, cv_kpts = sample_by_octave(cv_kpts, n_sample)

    @torchsnooper.snoop()
    def test_grayscale(self):
        image = cv2.imread('/Database/Test/abnormal_detection/test.jpg')
        gray_image = grayscale(image)

        cv2.imwrite('/Database/Test/abnormal_detection/test_gray_image.jpg', gray_image)

    @torchsnooper.snoop()
    def test_normalize_(self):
        pass

    @torchsnooper.snoop()
    def test_crop_image(self):
        image = cv2.imread('/Database/Test/abnormal_detection/test.jpg')
        size = (image.shape[0], image.shape[1])
        center = (50, 50)
        cropped_image, border, offset = crop_image(image, center, size)
        cv2.imwrite('/Database/Test/abnormal_detection/test_cropped_image.jpg', cropped_image)

    @torchsnooper.snoop()
    def test_cv_img_to_pil(self):
        image = cv2.imread('/Database/Test/abnormal_detection/test.jpg')
        pil_img = cv_img_to_pil(image)
        pil_img.save('/Database/Test/abnormal_detection/test_pil_image.jpg', 'jpeg')

    @torchsnooper.snoop()
    def test_cv_img_to_pil(self):
        from PIL import Image
        pil_image = Image.open('/Database/Test/abnormal_detection/test_pil_image.jpg')
        image = pil_img_to_cv(pil_image)
        cv2.imwrite('/Database/Test/abnormal_detection/test_cv_image.jpg', image)

    @torchsnooper.snoop()
    def test_cv_resize_img(self):
        image = cv2.imread('/Database/Test/abnormal_detection/test.jpg')
        resize_image = cv_resize_img(image)
        cv2.imwrite('/Database/Test/abnormal_detection/test_resize_image.jpg', resize_image)

        resize_image_200 = cv_resize_img(image, 200)
        cv2.imwrite('/Database/Test/abnormal_detection/test_resize_image_200.jpg', resize_image_200)


if __name__ == '__main__':
    unittest.main()
