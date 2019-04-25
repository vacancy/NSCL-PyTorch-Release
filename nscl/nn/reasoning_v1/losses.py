#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/04/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import os
import torch
import torch.nn.functional as F

from jacinle.utils.enum import JacEnum
from nscl.nn.losses import MultitaskLossBase
from nscl.datasets.definition import gdef

DEBUG_SCENE_LOSS = int(os.getenv('DEBUG_SCENE_LOSS', '0'))


__all__ = ['SceneParsingLoss', 'QALoss', 'ParserV1Loss']


class SceneParsingLoss(MultitaskLossBase):
    def __init__(self, used_concepts, add_supervision=False):
        super().__init__()
        self.used_concepts = used_concepts
        self.add_supervision = add_supervision

    def forward(self, feed_dict, f_sng, attribute_embedding, relation_embedding):
        outputs, monitors = dict(), dict()

        objects = [f[1] for f in f_sng]
        all_f = torch.cat(objects)

        for attribute, concepts in self.used_concepts['attribute'].items():
            if 'attribute_' + attribute not in feed_dict:
                continue

            all_scores = []
            for v in concepts:
                this_score = attribute_embedding.similarity(all_f, v)
                all_scores.append(this_score)

            all_scores = torch.stack(all_scores, dim=-1)
            all_labels = feed_dict['attribute_' + attribute]

            if all_labels.dim() == all_scores.dim() - 1:
                acc_key = 'acc/scene/attribute/' + attribute
                monitors[acc_key] = (
                    ((all_scores > 0).float().sum(dim=-1) == 1) *
                    (all_scores.argmax(-1) == all_labels.long())
                ).float().mean()

                if self.training and self.add_supervision:
                    this_loss = self._sigmoid_xent_loss(all_scores, all_labels.long())
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/attribute/' + attribute, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss
            else:
                acc_key = 'acc/scene/attribute/' + attribute
                monitors[acc_key] = (
                    (all_scores > 0).long() == all_labels.long()
                ).float().mean()

                if self.training and self.add_supervision:
                    this_loss = self._bce_loss(all_scores, all_labels.float())
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/attribute/' + attribute, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss

            if 'attribute_relation_' + attribute not in feed_dict:
                continue

            cross_scores = []
            for f in f_sng:
                cross_scores.append(attribute_embedding.cross_similarity(f[1], attribute).view(-1))
            cross_scores = torch.cat(cross_scores)
            cross_labels = feed_dict['attribute_relation_' + attribute]

            acc_key = 'acc/scene/attribute-relation/' + attribute
            monitors[acc_key] = ((cross_scores > 0).long() == cross_labels.long()).float().mean()

            # TODO(Jiayuan Mao @ 07/31): clean up these lines if we have confirmed that we don't need this.
            if self.training and self.add_supervision:
                this_loss = self._bce_loss(cross_scores, cross_labels.float())
                if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                    print('NAN! in object_same_loss. Starting the debugger')
                    from IPython import embed; embed()
                for loss_key in ['loss/scene/attribute-relation/' + attribute, 'loss/scene']:
                    monitors[loss_key] = monitors.get(loss_key, 0) + this_loss

        if len(self.used_concepts['relation']) > 0:
            relations = [f[2].view(-1, f[2].size(2)) for f in f_sng]
            all_f = torch.cat(relations)
            for relation, concepts in self.used_concepts['relation'].items():
                if 'relation_' + relation not in feed_dict:
                    continue

                all_scores = []
                for v in concepts:
                    this_score = relation_embedding.similarity(all_f, v)
                    all_scores.append(this_score)
                all_scores = torch.stack(all_scores, dim=-1)
                all_labels = feed_dict['relation_' + relation]

                acc_key = 'acc/scene/relation/' + relation
                monitors[acc_key] = ((all_scores > 0).long() == all_labels.long()).float().mean()

                if self.training and self.add_supervision:
                    this_loss = self._bce_loss(all_scores, all_labels.float())
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in relation_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/relation/' + relation, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss

        return monitors, outputs


class QALoss(MultitaskLossBase):
    def __init__(self, add_supervision):
        super().__init__()
        self.add_supervision = add_supervision

    def forward(self, feed_dict, answers, question_index=None, loss_weights=None, accuracy_weights=None):
        """
        Args:
            feed_dict (dict): input feed dict.
            answers (list): answer derived from the reasoning module.
            question_index (list[int]): question index of the i-th answer.
            loss_weights (list[float]):
            accuracy_weights (list[float]):

        """

        monitors = {}
        outputs = {'answer': []}

        if 'answer' not in feed_dict or 'question_type' not in feed_dict:
            return monitors, outputs

        for i, (query_type, a) in enumerate(answers):
            j = i if question_index is None else question_index[i]
            loss_w = loss_weights[i] if loss_weights is not None else 1
            acc_w = accuracy_weights[i] if accuracy_weights is not None else 1

            gt = feed_dict['answer'][j]
            response_query_type = gdef.qtype2atype_dict[query_type]

            question_type = feed_dict['question_type'][j]
            response_question_type = gdef.qtype2atype_dict[question_type]

            if response_question_type != response_query_type:
                key = 'acc/qa/' + query_type
                monitors.setdefault(key, []).append((0, acc_w))
                monitors.setdefault('acc/qa', []).append((0, acc_w))

                if self.training and self.add_supervision:
                    l = torch.tensor(10, dtype=torch.float, device=a[0].device if isinstance(a, tuple) else a.device)
                    monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                    monitors.setdefault('loss/qa', []).append((l, loss_w))
                continue

            if response_query_type == 'word':
                a, word2idx = a
                argmax = a.argmax(dim=-1).item()
                idx2word = {v: k for k, v in word2idx.items()}
                outputs['answer'].append(idx2word[argmax])
                gt = word2idx[gt]
                loss = self._xent_loss
            elif response_query_type == 'bool':
                argmax = int((a > 0).item())
                outputs['answer'].append(argmax)
                gt = int(gt)
                loss = self._bce_loss
            elif response_query_type == 'integer':
                try:
                    argmax = int(round(a.item()))
                except ValueError:
                    argmax = 0
                outputs['answer'].append(argmax)
                gt = int(gt)
                loss = self._mse_loss
            else:
                raise ValueError('Unknown query type: {}.'.format(response_query_type))

            key = 'acc/qa/' + query_type
            monitors.setdefault(key, []).append((int(gt == argmax), acc_w))
            monitors.setdefault('acc/qa', []).append((int(gt == argmax), acc_w))

            if self.training and self.add_supervision:
                l = loss(a, gt)
                monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                monitors.setdefault('loss/qa', []).append((l, loss_w))

        return monitors, outputs

    def _gen_normalized_weights(self, weights, n):
        if weights is None:
            return [1 for _ in range(n)]
        sum_weights = sum(weights)
        return [w / sum_weights * n]


class ParserV1RewardShape(JacEnum):
    LOSS = 'loss'
    ACCURACY = 'accuracy'


class ParserV1Loss(MultitaskLossBase):
    def __init__(self, reward_shape='loss'):
        super().__init__()
        self.reward_shape = ParserV1RewardShape.from_string(reward_shape)

    def forward(self, feed_dict, programs_pd, accuracy, loss):
        batch_size = len(programs_pd)
        policy_loss = 0
        for i in range(len(feed_dict['question_raw'])):
            log_likelihood = [p['log_likelihood'] for p in programs_pd if i == p['scene_id']]
            if len(log_likelihood) == 0:
                continue
            log_likelihood = torch.stack(log_likelihood, dim=0)
            discounted_log_likelihood = [p['discounted_log_likelihood'] for p in programs_pd if i == p['scene_id']]
            discounted_log_likelihood = torch.stack(discounted_log_likelihood, dim=0)

            if self.reward_shape is ParserV1RewardShape.LOSS:
                # reward = -loss
                rewards = 10 - torch.stack([loss[j] for j, p in enumerate(programs_pd) if i == p['scene_id']], dim=0)
                likelihood = F.softmax(log_likelihood, dim=-1)
            elif self.reward_shape is ParserV1RewardShape.ACCURACY:
                rewards = torch.tensor([accuracy[j] for j, p in enumerate(programs_pd) if i == p['scene_id']]).to(discounted_log_likelihood)
                likelihood = F.softmax(log_likelihood * rewards + -1e6 * (1 - rewards), dim=-1)

            # \Pr[p] * reward * \nabla \log \Pr[p]
            policy_loss += (-(likelihood * rewards).detach() * discounted_log_likelihood).sum()
        return {'loss/program': policy_loss}, dict()

