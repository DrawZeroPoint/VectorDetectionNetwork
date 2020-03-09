#!/usr/bin/env python
# coding=utf-8


import unittest

import modules.vdn as vdn


class Test(unittest.TestCase):

    # def test_eval_default(self):
    #     """
    #     """
    #     vdn_instance = vdn.VectorDetectionNetwork(backbone='resnet50')
    #     vdn_instance.eval()

    def test_eval_resnet34(self):
        vdn_instance = vdn.VectorDetectionNetwork(backbone='resnet34')
        vdn_instance.eval()


if __name__ == '__main__':
    unittest.main()
