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
        vis_util.save_np_heatmaps(batch_heatmaps, "/VDN/data/test/test_heatmap_all.jpg")

    @staticmethod
    def generate_gaussian(heatmap, mu_x, mu_y, tmp_size=9, scale=1.0):
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2

        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * 3 ** 2))
        g_x = max(0, -ul[0]), min(br[0], 96) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], 96) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], 96)
        img_y = max(0, ul[1]), min(br[1], 96)
        prev_val = heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]]
        curr_val = g[g_y[0]:g_y[1], g_x[0]:g_x[1]] * scale
        combined_val = np.maximum(prev_val, curr_val)
        # print(f'com {combined_val}')
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = combined_val

    def test_get_local_maximum(self):
        heatmap = np.zeros((96, 96))

        self.generate_gaussian(heatmap, 30, 30)
        self.generate_gaussian(heatmap, 30, 40, scale=0.8)

        peaks, peakvals = lib_inference.get_peaks_by_local_maximum(heatmap)
        print(peaks, peakvals)

        vis_util.save_np_heatmaps(heatmap, "/VDN/data/test/test_get_local_maximum.jpg")

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
