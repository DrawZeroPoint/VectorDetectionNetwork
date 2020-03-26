#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import os
import cv2
import unittest
import torchsnooper

from utils.cv.util import *


class Test(unittest.TestCase):

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
