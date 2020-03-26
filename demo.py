#!/usr/bin/env python
# coding=utf-8

import cv2
import os
import unittest

import modules.vdn as vdn


class Test(unittest.TestCase):

    def test_demo_default(self):
        VDN = vdn.VectorDetectionNetwork(backbone='resnet50')

        demo_data_dir = "./data/demo"
        if not os.path.exists(demo_data_dir):
            raise FileNotFoundError(f'{demo_data_dir} not exist')

        file_list = os.listdir(demo_data_dir)
        file_num = len(file_list)

        if not file_num:
            print('No available image in data/demo')
            return

        print(f'Got {file_num} image(s) for demo')

        total_spent = 0
        cnt = 0
        for k, item in enumerate(file_list):
            image_path = os.path.join(demo_data_dir, item)
            src_img = cv2.imread(image_path)

            print(f'Result of image {k+1}: {item}')
            _, _, _, spent = VDN.get_vectors(src_img, verbose=item[:-4])

            if k > 0:
                # The first image is not counted due to loading time
                total_spent += spent
                cnt += 1

        if total_spent > 0:
            print('inference rate (fps): ', cnt/total_spent)


if __name__ == '__main__':
    unittest.main()
