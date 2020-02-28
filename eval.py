#!/usr/bin/env python
# coding=utf-8


import unittest

import modules.vdn as vdn


class Test(unittest.TestCase):

    def test_eval(self):
        """
        """
        vdn_instance = vdn.VectorDetectionNetwork()
        vdn_instance.eval()


if __name__ == '__main__':
    unittest.main()
