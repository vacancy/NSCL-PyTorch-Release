#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/04/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import jactorch

__all__ = ['SigmoidCrossEntropy', 'MultilabelSigmoidCrossEntropy']


class SigmoidCrossEntropy(nn.Module):
    def __init__(self, one_hot=False):
        super().__init__()
        self.one_hot = one_hot
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        if not self.one_hot:
            target = jactorch.one_hot_nd(target, input.size(-1))
        return self.bce(input, target).sum(dim=-1).mean()


class MultilabelSigmoidCrossEntropy(nn.Module):
    def __init__(self, one_hot=False):
        super().__init__()
        self.one_hot = one_hot
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, labels):
        if type(labels) in (tuple, list):
            labels = torch.tensor(labels, dtype=torch.int64, device=input.device)

        assert input.dim() == 1
        if not self.one_hot:
            with torch.no_grad():
                mask = torch.zeros_like(input)
                if labels.size(0) > 0:
                    ones = torch.ones_like(labels, dtype=torch.float32)
                    mask.scatter_(0, labels, ones)
            labels = mask

        return self.bce(input, labels).sum(dim=-1).mean()


class MultitaskLossBase(nn.Module):
    def __init__(self):
        super().__init__()

        self._sigmoid_xent_loss = SigmoidCrossEntropy()
        self._multilabel_sigmoid_xent_loss = MultilabelSigmoidCrossEntropy()

    def _mse_loss(self, pred, label):
        return (pred - label).abs()

    def _bce_loss(self, pred, label):
        return -( jactorch.log_sigmoid(pred) * label + jactorch.log_sigmoid(-pred) * (1 - label) ).mean()

    def _xent_loss(self, pred, label):
        logp = F.log_softmax(pred, dim=-1)
        return -logp[label].mean()
