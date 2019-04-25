#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : object_repr.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/09/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn

from . import functional


class ObjectBasedRepresentation(nn.Module):
    def __init__(self, feature_dim, downsample_rate, pool_size=7):
        super().__init__()
        self.pool_size = pool_size
        self.feature_dim = feature_dim
        self.downsample_rate = downsample_rate

        self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

    def forward(self, input, objects, objects_length):
        object_features = input
        context_features = input

        outputs = list()
        objects_index = 0
        for i in range(input.size(0)):
            box = objects[objects_index:objects_index + objects_length[i].item()]
            objects_index += objects_length[i].item()

            with torch.no_grad():
                batch_ind = i + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)

                # generate a "full-image" bounding box
                image_h, image_w = input.size(2) * self.downsample_rate, input.size(3) * self.downsample_rate
                image_box = torch.cat([
                    torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                    torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                    image_w + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                    image_h + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)
                ], dim=-1)

                # intersection maps
                box_context_imap = functional.generate_intersection_map(box, image_box, self.pool_size)

            this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1)[:1])
            this_object_features = torch.cat([
                self.object_roi_pool(object_features, torch.cat([batch_ind, box], dim=-1)),
            ], dim=1)

            outputs.append((this_object_features, this_context_features[0], box_context_imap))

        return outputs

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)

