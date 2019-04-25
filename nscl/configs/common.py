#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : common.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/29/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Common configuration.
"""

from jacinle.utils.container import G

__all__ = ['make_base_configs']


class Config(G):
    pass


def make_base_configs():
    configs = Config()

    configs.data = G()
    configs.model = G()
    configs.train = G()
    configs.train.weight_decay = 1e-4

    return configs

