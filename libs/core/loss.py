#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsnooper


# @torchsnooper.snoop()
def _gather_feat(feat, ind, mask=None):
    """

    :param feat: (b, 96*96, 2)  2 for vx, vy respectively
    :param ind: (b, j, k)
    :param mask: None
    :return: (b, k, 2)
    """
    dim = feat.size(2)  # 2
    ind = ind.squeeze(1)  # (b, k)
    ind = ind.expand(ind.size(0), dim, ind.size(1))  # (b, 2, k)
    feat = feat.permute(0, 2, 1)  # (b, 2, 96*96)
    feat = feat.gather(2, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat.permute(0, 2, 1)


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, out_vector, target_vector, tgt_indexes):
        """
        :param out_vector: torch.Size([b, 2, 96, 96])
        :param target_vector: torch.Size([b, j, k, 2])
        :param tgt_indexes: torch.Size([b, j, k])
        :return:
        """
        pred_vector = _transpose_and_gather_feat(out_vector, tgt_indexes)  # (b, k ,2)
        loss = F.l1_loss(pred_vector, target_vector.squeeze(1), reduction='sum')
        loss = loss / (pred_vector.size(1) + 1e-4)
        # print(f'---- Object number {pred_vector.size(1)} Loss {loss}')
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, target):
        """

        :param output: [b, 2, 96, 96]
        :param target: [b, j, 2, 96, 96]
        :return:
        """
        loss = F.l1_loss(output, target.squeeze(1))
        return loss


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    # @torchsnooper.snoop()
    def forward(self, out_heatmap, target_heatmap):
        """
        :param out_heatmap: torch.Size([b, 1, 96, 96])
        :param target_heatmap: torch.Size([b, 1, 96, 96])
        :return:
        """
        batch_size = out_heatmap.size(0)
        num_joints = out_heatmap.size(1)  # 1 by default

        '''
        Here split(1, 1) turns the tensor with dim (b_size, n_joints, w*h) to a tuple of tensor:
        (<(b_size, 1, w*h)>, ..., <(b_size, 1, w*h)>), which contains num_joints elements.
        '''
        heatmaps_pred = out_heatmap.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target_heatmap.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0
        for j in range(num_joints):
            heatmap_pred = heatmaps_pred[j].squeeze()
            heatmap_gt = heatmaps_gt[j].squeeze()

            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
