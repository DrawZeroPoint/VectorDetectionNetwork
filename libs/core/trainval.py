#!/usr/bin/env python
# encoding: utf-8

import logging
import time
import os

import numpy as np
import torch

import libs.core.inference as lib_inference
from libs.core.config import get_model_name
from libs.core.evaluate import accuracy
from libs.utils.vis import save_debug_images
from libs.utils.utils import vector_components_to_deg


logger = logging.getLogger(__name__)


def train(config, train_loader, model, crit_heatmap, crit_vector, optimizer, epoch, output_dir, writer_dict=None):
    end_epoch = float(config.TRAIN.END_EPOCH)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_heatmap, target_vectormap, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        out_heatmap, out_vector = model(input)

        # Set non_blocking=True is the standard operation before BP
        target_heatmap = target_heatmap.cuda(non_blocking=True)
        target_vectormap = target_vectormap.cuda(non_blocking=True)

        # print(f'shape of out heatmap {out_heatmap.shape}, out_vector {out_vector.shape},'
        #       f'target_heatmap {target_heatmap.shape}, target_vector {target_vector.shape},'
        #       f'tgt_indexes {tgt_indexes.shape}')

        j_loss = crit_heatmap(out_heatmap, target_heatmap)
        v_loss = crit_vector(out_vector, target_vectormap.squeeze(1))
        loss = j_loss + (0.001 + epoch * 0.01 / end_epoch) * v_loss
        # if epoch < 50:
        #     loss = j_loss
        # elif 50 <= epoch < 100:
        #     loss = j_loss + (epoch - 50) / 50 * v_loss
        # else:
        #     loss = j_loss + v_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred_j, pred_v = accuracy(
            out_heatmap.detach().cpu().numpy(),
            out_vector.detach().cpu().numpy(),
            target_heatmap.detach().cpu().numpy(),
            target_vectormap.squeeze(1).detach().cpu().numpy()  # reduce the joint dim (=1)
        )

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            print(f'jloss: {j_loss.item()}, vloss: {v_loss.item()}')

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target_heatmap, pred_j, pred_v,
                              out_heatmap, out_vector, prefix)


def validate(config, val_loader, val_dataset, model, crit_heatmap, crit_vector, epoch, output_dir, writer_dict=None):
    end_epoch = float(config.TRAIN.END_EPOCH)
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    image_path = []
    filenames = []
    imgnums = []
    max_instance_num = 3  # how many instances could be in one sample image

    all_kp_preds = None  # keypoint location kp_preds for OKS metric
    all_vd_preds = None  # vector direction kp_preds for VDS metric
    all_boxes = None

    with torch.no_grad():
        end = time.time()

        for i, (input, target_heatmap, target_vectormap, meta) in enumerate(val_loader):
            
            # compute output
            out_hm, out_vm = model(input)

            target_heatmap = target_heatmap.cuda(non_blocking=True)
            target_vectormap = target_vectormap.cuda(non_blocking=True)

            j_loss = crit_heatmap(out_hm, target_heatmap)
            v_loss = crit_vector(out_vm, target_vectormap.squeeze(1))

            # if epoch < 50:
            #     loss = j_loss
            # elif 50 <= epoch < 100:
            #     loss = j_loss + (epoch - 50) / 50 * v_loss
            # else:
            #     loss = j_loss + v_loss

            loss = j_loss + epoch / end_epoch * v_loss

            num_images = input.size(0)  # aka, batch size
            losses.update(loss.item(), num_images)

            _, avg_acc, cnt, pred_j, pred_v = accuracy(
                out_hm.detach().cpu().numpy(),
                out_vm.detach().cpu().numpy(),
                target_heatmap.detach().cpu().numpy(),
                target_vectormap.squeeze(1).detach().cpu().numpy()  # reduce the joint dim (=1)
            )

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()  # default 1

            j_preds, _, v_preds, maxvals = lib_inference.get_final_preds(out_hm.clone().cpu().numpy(),
                                                                         out_vm.clone().cpu().numpy(), c, s)

            # sort the predictions and get the first 3 with highest score
            # k is the instance number predicted, only get the first 3 instances if k > max_instance_num
            # joint_preds (b, j, k, 2), maxvals (b, j, k, 1)
            # sorted_joint_preds = joint_preds
            # sorted_maxvals = maxvals
            # sorted_joint_preds = sort_multi_dimension_array(joint_preds, maxvals, 2)
            # sorted_maxvals = -np.sort(-maxvals, 2)  # in descending order

            if j_preds.ndim != 4 or j_preds.shape[-1] != 2:
                print('=> invalid joint prediction')
                continue

            det_num = min(j_preds.shape[-2], max_instance_num)
            for m in range(det_num):
                js = j_preds.shape
                det_j_pred = np.zeros((js[0], js[1], 3, 2))
                det_v_pred = np.zeros((js[0], js[1], 3, 2))
                det_val = np.zeros((js[0], js[1], 3, 1))
                det_j_pred[:, :, 0] = j_preds[:, :, m]
                det_v_pred[:, :, 0] = v_preds[:, :, m]
                det_val[:, :, 0] = maxvals[:, :, m]

                det_j_pred = np.squeeze(det_j_pred, 1)  # squeeze the joint dim cause joint_num=1
                det_v_pred = np.squeeze(det_v_pred, 1)
                det_val = np.squeeze(det_val, 1)

                if det_j_pred.shape[-1] == 2 and det_val.shape[-1] == 1:
                    if all_kp_preds is None:
                        all_kp_preds = np.zeros((num_images, 3, 3))
                        all_kp_preds[:, 0:3, 0:2] = det_j_pred[:, 0:3, :]
                        all_kp_preds[:, 0:3, 2:3] = det_val[:, 0:3, :]
                    else:
                        tmp_preds = np.zeros((num_images, 3, 3))
                        tmp_preds[:, 0:3, 0:2] = det_j_pred[:, 0:3, :]
                        tmp_preds[:, 0:3, 2:3] = det_val[:, 0:3, :]
                        all_kp_preds = np.concatenate((all_kp_preds, tmp_preds), axis=0)

                    det_ang_pred = vector_components_to_deg(det_v_pred)
                    if all_vd_preds is None:
                        all_vd_preds = np.ones((num_images, 3, 3))
                        all_vd_preds[:, 0:3, :] = det_ang_pred[:, 0:3, :]
                    else:
                        tmp_preds = np.zeros((num_images, 3, 3))
                        tmp_preds[:, :, :] = det_ang_pred[:, :, :]
                        all_vd_preds = np.concatenate((all_vd_preds, tmp_preds), axis=0)

                if all_boxes is None:
                    all_boxes = np.zeros((num_images, 6))
                    all_boxes[:, 0:2] = c[:, 0:2]
                    all_boxes[:, 2:4] = s[:, 0:2]
                    all_boxes[:, 4] = np.prod(s * 200, 1)
                    all_boxes[:, 5] = score
                else:
                    tmp_boxes = np.zeros((num_images, 6))
                    tmp_boxes[:, 0:2] = c[:, 0:2]
                    tmp_boxes[:, 2:4] = s[:, 0:2]
                    tmp_boxes[:, 4] = np.prod(s * 200, 1)
                    tmp_boxes[:, 5] = score
                    all_boxes = np.concatenate((all_boxes, tmp_boxes), axis=0)

                image_path.extend(meta['image'])

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)
                print(f'jloss: {j_loss.item()}, vloss: {v_loss.item()}')

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target_heatmap, pred_j, pred_v,
                                  out_hm, out_vm, prefix)

        oks_metric, vds_metric, perf_indicator = val_dataset.evaluate(config, all_kp_preds, all_vd_preds,
                                                                      output_dir, all_boxes,
                                                                      image_path, filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(oks_metric, list):
            for name_value in oks_metric:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(oks_metric, full_arch_name)

        if isinstance(vds_metric, list):
            for name_value in vds_metric:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(vds_metric, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(oks_metric, list):
                for name_value in oks_metric:
                    writer.add_scalars('valid_oks', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid_oks', dict(oks_metric), global_steps)
            if isinstance(vds_metric, list):
                for name_value in vds_metric:
                    writer.add_scalars('valid_vds', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid_vds', dict(vds_metric), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) + ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
