#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/06/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch

__all__ = ['canonize_monitors', 'update_from_loss_module']


def canonize_monitors(monitors):
    for k, v in monitors.items():
        if isinstance(monitors[k], list):
            if isinstance(monitors[k][0], tuple) and len(monitors[k][0]) == 2:
                monitors[k] = sum([a * b for a, b in monitors[k]]) / max(sum([b for _, b in monitors[k]]), 1e-6)
            else:
                monitors[k] = sum(v) / max(len(v), 1e-3)
        if isinstance(monitors[k], float):
            monitors[k] = torch.tensor(monitors[k])


def update_from_loss_module(monitors, output_dict, loss_update):
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)

