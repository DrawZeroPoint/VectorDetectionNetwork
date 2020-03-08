#!/usr/bin/env python
# coding=utf-8


import unittest

import modules.vdn as vdn


class Test(unittest.TestCase):

    def test_train_default(self):
        """
        """
        vdn_instance = vdn.VectorDetectionNetwork(train=True, backbone='resnet50')
        vdn_instance.train()


if __name__ == '__main__':
    unittest.main()
