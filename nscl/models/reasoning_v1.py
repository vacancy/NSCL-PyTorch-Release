#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : reasoning_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/06/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch.nn as nn
import jactorch.nn as jacnn

from jacinle.logging import get_logger
from nscl.configs.common import make_base_configs
from nscl.datasets.definition import gdef

logger = get_logger(__file__)

__all__ = ['make_reasoning_v1_configs', 'ReasoningV1Model']


def make_reasoning_v1_configs():
    configs = make_base_configs()

    # data configs
    configs.data.image_size = 256
    configs.data.add_full_image_bbox = False

    # model configs for scene graph
    configs.model.sg_dims = [None, 256, 256]

    # model ocnfigs for visual-semantic embeddings
    configs.model.vse_known_belong = False
    configs.model.vse_large_scale = False
    configs.model.vse_ls_load_concept_embeddings = False
    configs.model.vse_hidden_dims = [None, 64, 64]

    # model configs for parser
    configs.model.word_embedding_dim = 300
    configs.model.positional_embedding_dim = 50
    configs.model.word_embedding_dropout = 0.5
    configs.model.gru_dropout = 0.5
    configs.model.gru_hidden_dim = 256

    # supervision configs
    configs.train.discount = 0.9
    configs.train.scene_add_supervision = False
    configs.train.qa_add_supervision = False
    configs.train.parserv1_reward_shape = 'loss'

    return configs


class ReasoningV1Model(nn.Module):
    def __init__(self, vocab, configs):
        super().__init__()
        self.vocab = vocab

        import jactorch.models.vision.resnet as resnet
        self.resnet = resnet.resnet34(pretrained=True, incl_gap=False, num_classes=None)
        self.resnet.layer4 = jacnn.Identity()

        import nscl.nn.scene_graph.scene_graph as sng
        # number of channels = 256; downsample rate = 16.
        self.scene_graph = sng.SceneGraph(256, configs.model.sg_dims, 16)

        import nscl.nn.reasoning_v1.quasi_symbolic as qs
        self.reasoning = qs.DifferentiableReasoning(
            self._make_vse_concepts(configs.model.vse_large_scale, configs.model.vse_known_belong),
            self.scene_graph.output_dims, configs.model.vse_hidden_dims
        )

        import nscl.nn.reasoning_v1.losses as vqa_losses
        self.scene_loss = vqa_losses.SceneParsingLoss(gdef.all_concepts, add_supervision=configs.train.scene_add_supervision)
        self.qa_loss = vqa_losses.QALoss(add_supervision=configs.train.qa_add_supervision)

    def train(self, mode=True):
        super().train(mode)

    def _make_vse_concepts(self, large_scale, known_belong):
        if large_scale:
            return {
                'attribute_ls': {'attributes': list(gdef.ls_attributes), 'concepts': list(gdef.ls_concepts)},
                'relation_ls': {'attributes': None, 'concepts': list(gdef.ls_relational_concepts)},
                'embeddings': gdef.get_ls_concept_embeddings()
            }
        return {
            'attribute': {
                'attributes': list(gdef.attribute_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.attribute_concepts.items() for v in vs
                ]
            },
            'relation': {
                'attributes': list(gdef.relational_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.relational_concepts.items() for v in vs
                ]
            }
        }
