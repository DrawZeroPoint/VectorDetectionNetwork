#!/usr/bin/env python
# coding=utf-8


import os
import logging
import time

from pathlib2 import Path

import torch
import numpy as np
import torch.optim as optim

from libs.core.config import get_model_name


def create_logger(cfg, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating root output dir {}'.format(root_output_dir))
        root_output_dir.mkdir()

    model, _ = get_model_name(cfg)
    cfg_name = 'vdn_model'

    final_output_dir = root_output_dir

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / (''.join((cfg_name, '_', time_str)))
    print('=> creating tensor board log dir {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir):
    filename = states['model'][0]
    ckpt_name = '_'.join((filename, 'checkpoint.pth.tar'))
    best_name = '_'.join((filename, 'best.pth.tar'))
    # torch.save(states, os.path.join(output_dir, ckpt_name))
    if states['epoch'] % 40 == 0 and states['epoch'] != 0:
        torch.save(states['state_dict'], os.path.join(output_dir, best_name))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, best_name))


def sort_multi_dimension_array(sort_array: np.ndarray, ind_array: np.ndarray, axis: int):
    """Sort given array by another array in descending order

    :param sort_array: (b, j, k, 2)
    :param ind_array: (b, j, k, 1)
    :param axis:
    :return:
    """
    s_shape = sort_array.shape
    i_shape = ind_array.shape

    assert s_shape[axis] == i_shape[axis]

    try:
        result_list = list(map(lambda x, y: y[:][:][x], np.argsort(-ind_array, axis), sort_array))
        return result_list[0].reshape(s_shape)
    except IndexError:
        return sort_array


def vector_components_to_deg(cmp: np.ndarray):
    """This function is tailored for converting predicted vector components to degree
    fitted specific format to compute with coco eval

    :param cmp: (b, k, 2), 2 for vx, vy
    :return: (b, k, 3), 3 for (deg, deg, 1)
    """
    assert cmp.ndim == 3

    x_cmp = cmp[:, :, 0]
    y_cmp = cmp[:, :, 1]

    rad = np.arctan2(y_cmp, x_cmp)
    deg = np.rad2deg(rad)

    out = np.ones((cmp.shape[0], cmp.shape[1], 3))
    out[:, :, 0] = deg[0][0]
    out[:, :, 1] = deg[0][0]

    # print('==>', out)
    return out


if __name__ == "__main__":
    sort_arr = np.array([[[8, 5], [9, 6], [2, 2], [4, 3]]])
    ind_arr = np.array([[[1], [5], [4], [10]]])

    res = sort_multi_dimension_array(sort_arr, ind_arr, 1)
    print(res)
