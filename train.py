#!/usr/bin/env python
# coding=utf-8


import unittest

import modules.vdn as vdn


class Test(unittest.TestCase):

    def test_train(self):
        """Available backbones: resnet34 (default), resnet18, resnet50.

        """
        vdn_instance = vdn.VectorDetectionNetwork(train=True, backbone='resnet34')
        vdn_instance.train()


if __name__ == '__main__':
    unittest.main()
