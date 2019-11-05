#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn
import json
import torchsnooper


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    # @torchsnooper.snoop()
    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)

        '''
        Here split(1, 1) turns the tensor with dim (b_size, n_joints, w*h) to a tuple of tensor:
        (<(b_size, 1, w*h)>, ..., <(b_size, 1, w*h)>), which contains 5 elements.
        '''
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
