#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import cv2
import os
import unittest
import numpy as np

import utils.vis.util as vis_util
import libs.core.inference as lib_inference


class Test(unittest.TestCase):

    def test_get_all_preds(self):
        batch_heatmaps = np.load("/VDN/data/test/test.npy")
        vis_util.save_np_heatmaps(batch_heatmaps, "/VDN/data/test/test_heatmap_all.jpg", is_max=False)

    # def test_swapaxes(self):
    #     testdata = np.array([[[149, 243,  2],
    #                           [226, 198,  2],
    #                           [158, 184,  2]],
    #                          [[223,  77,  2],
    #                           [302, 214,  2],
    #                           [59, 146, 2]]])
    #     print(testdata.shape)
    #     testdata = np.swapaxes(testdata, 0, 1)
    #     print(testdata)
    #     print(testdata.shape)


if __name__ == '__main__':
    unittest.main()
