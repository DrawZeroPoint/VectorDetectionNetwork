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
        VDN = vdn.VectorDetectionNetwork()
        VDN.train('vdn_model')


if __name__ == '__main__':
    unittest.main()
