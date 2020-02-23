#!/usr/bin/env python
# coding=utf-8


import cv2
import os
import unittest

import modules.vdn as vdn


class Test(unittest.TestCase):

    def test_demo(self):
        """
        """
        VDN = vdn.VectorDetectionNetwork()

        demo_data_dir = "./data/demo"
        file_list = os.listdir(demo_data_dir)
        file_num = len(file_list)

        if not file_num:
            print('No available image in data/demo')
            return

        print(f'Got {file_num} image(s) for demo')

        for k, item in enumerate(file_list):
            image_path = os.path.join(demo_data_dir, item)
            src_img = cv2.imread(image_path)

            VDN.get_vectors(src_img, verbose=k+1)


if __name__ == '__main__':
    unittest.main()
