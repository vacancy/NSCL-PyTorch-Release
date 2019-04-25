#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_graph_groundtruth.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/31/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import jactorch

__all__ = ['SceneGraphGroundtruth']


class SceneGraphGroundtruth(nn.Module):
    def __init__(self, vocab, used_concepts):
        super().__init__()
        self.vocab = vocab
        self.used_concepts = used_concepts

        self.output_dims = [None, 0, 4]

        self.register_buffer('global2local', torch.zeros(len(self.vocab), dtype=torch.int64))
        for k, v in self.used_concepts.items():
            if v['type'] != 'attribute':
                continue

            self.output_dims[1] += len(v['values'])

            v = v['values']
            self.register_buffer('local2global_{}'.format(k), torch.zeros(len(v), dtype=torch.int64))
            for i, vv in enumerate(v):
                self.global2local[vocab.word2idx[vv]] = i
                getattr(self, 'local2global_{}'.format(k))[i] = vocab.word2idx[vv]

    def forward(self, input, objects, objects_length, feed_dict):
        # note that we will just ignore the input LOL

        objects_index = 0
        relation_index = 0
        outputs = []
        for i in range(input.size(0)):

            nr_objects = objects_length[i].item()
            object_features = []
            for attribute, info in self.used_concepts.items():
                if info['type'] == 'attribute':
                    values = feed_dict['objects_' + attribute][objects_index:objects_index+nr_objects]
                    mapped_values = self._valmap(self.global2local, values)
                    object_features.append(jactorch.one_hot(mapped_values, len(info['values'])))

            object_features = torch.cat(object_features, dim=-1)
            object_features = object_features.float().to(input.device)
            relation_features = feed_dict['relations_spatial_relation'][relation_index:relation_index + nr_objects*nr_objects]

            relation_features = relation_features.float().to(input.device)
            outputs.append((None, object_features, relation_features.view(nr_objects, nr_objects, 4)))

            objects_index += nr_objects
            relation_index += nr_objects * nr_objects

        return outputs

    @staticmethod
    def _valmap(a, i):
        return a[i.view(-1)].view_as(i)
