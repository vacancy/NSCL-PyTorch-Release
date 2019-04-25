#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/19/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch
import jactorch

__all__ = [
    'box_size', 'box_intersection', 'box_iou',
    'generate_union_box', 'generate_roi_pool_bins', 'generate_intersection_map'
]


COOR_TO_LEN_CORR = 0


def __last(arr, x):
    return arr.narrow(-1, x, 1).squeeze(-1)


def box_size(box, c2l=COOR_TO_LEN_CORR):
    return (__last(box, 2) - __last(box, 0) + c2l) * (__last(box, 3) - __last(box, 1) + c2l)


def box_intersection(box1, box2, ratio=False, c2l=COOR_TO_LEN_CORR):
    xmin, ymin = [torch.max(__last(box1, i), __last(box2, i)) for i in range(2)]
    xmax, ymax = [torch.min(__last(box1, i), __last(box2, i)) for i in range(2, 4)]
    iw = torch.max(xmax - xmin + c2l, torch.zeros_like(xmax))
    ih = torch.max(ymax - ymin + c2l, torch.zeros_like(ymax))
    inter = iw * ih
    if ratio:
        return inter / box_size(box2)
    return inter


def box_iou(box1, box2):
    inter = box_intersection(box1, box2)
    union = box_size(box1) + box_size(box2) - inter
    return inter / union


def generate_union_box(box1, box2):
    xmin, ymin = [torch.min(__last(box1, i), __last(box2, i)) for i in range(2)]
    xmax, ymax = [torch.max(__last(box1, i), __last(box2, i)) for i in range(2, 4)]
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def generate_roi_pool_bins(box, bin_size, c2l=COOR_TO_LEN_CORR):
    # TODO(Jiayuan Mao @ 07/20): workaround: line space is not implemented for cuda.
    linspace = torch.linspace(0, 1, bin_size + 1, dtype=box.dtype).to(device=box.device)
    for i in range(box.dim() - 1):
        linspace.unsqueeze_(0)
    x_space = linspace * (__last(box, 2) - __last(box, 0) + c2l).unsqueeze(-1) + __last(box, 0).unsqueeze(-1)
    y_space = linspace * (__last(box, 3) - __last(box, 1) + c2l).unsqueeze(-1) + __last(box, 1).unsqueeze(-1)
    x1, x2 = x_space[:, :-1], x_space[:, 1:] - c2l
    y1, y2 = y_space[:, :-1], y_space[:, 1:] - c2l
    y1, x1 = jactorch.meshgrid(y1, x1, dim=-1)
    y2, x2 = jactorch.meshgrid(y2, x2, dim=-1)

    # shape: nr_boxes, bin_size^2, 4
    bins = torch.stack([x1, y1, x2, y2], dim=-1).view(box.size(0), -1, 4)
    return bins.float()


def generate_intersection_map(box1, box2, bin_size, c2l=COOR_TO_LEN_CORR):
    # box: nr_boxes, 4
    # bins: nr_boxes, bin_size^2, 4
    bins = generate_roi_pool_bins(box2, bin_size, c2l)
    box1 = box1.unsqueeze(1).expand_as(bins)
    return box_intersection(box1, bins, ratio=True, c2l=c2l).view(box1.size(0), 1, bin_size, bin_size).float()

