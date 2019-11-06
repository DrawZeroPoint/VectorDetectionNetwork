#!/usr/bin/env python
# coding=utf-8


import cv2
import os
import unittest
import torchsnooper

import modules.vdn as vdn


class Test(unittest.TestCase):

    def test_train(self):
        """
        """
        vdn_instance = vdn.VectorDetectionNetwork(train=True)
        vdn_instance.train()


if __name__ == '__main__':
    unittest.main()
