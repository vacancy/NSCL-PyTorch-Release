#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quasi_symbolic.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.


"""
Quasi-Symbolic Reasoning.
"""

import six

import torch
import torch.nn as nn

import jactorch.nn.functional as jacf

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from nscl.datasets.common.program_executor import ParameterResolutionMode
from nscl.datasets.definition import gdef
from . import concept_embedding, concept_embedding_ls
from . import quasi_symbolic_debug

logger = get_logger(__file__)

__all__ = ['ConceptQuantizationContext', 'ProgramExecutorContext', 'DifferentiableReasoning', 'set_apply_self_mask']


_apply_self_mask = {'relate': True, 'relate_ae': True}


def set_apply_self_mask(key, value):
    logger.warning('Set {}.apply_self_mask[{}] to {}.'.format(set_apply_self_mask.__module__, key, value))
    assert key in _apply_self_mask, key
    _apply_self_mask[key] = value


def do_apply_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return m * (1 - self_mask) + (-10) * self_mask


class InferenceQuantizationMethod(JacEnum):
    NONE = 0
    STANDARD = 1
    EVERYTHING = 2


_test_quantize = InferenceQuantizationMethod.STANDARD


def set_test_quantize(mode):
    global _test_quantize
    _test_quantize = InferenceQuantizationMethod.from_string(mode)



class ConceptQuantizationContext(nn.Module):
    def __init__(self, attribute_taxnomy, relation_taxnomy, training=False, quasi=False):
        """
        Args:
            attribute_taxnomy: attribute-level concept embeddings.
            relation_taxnomy: relation-level concept embeddings.
            training (bool): training mode or not.
            quasi(bool): if False, quantize the results as 0/1.

        """

        super().__init__()

        self.attribute_taxnomy = attribute_taxnomy
        self.relation_taxnomy = relation_taxnomy
        self.quasi = quasi

        super().train(training)

    def forward(self, f_sng):
        batch_size = len(f_sng)
        output_list = [dict() for i in range(batch_size)]

        for i in range(batch_size):
            f = f_sng[i][1]
            nr_objects = f.size(0)

            output_list[i]['filter'] = dict()
            for concept in self.attribute_taxnomy.all_concepts:
                scores = self.attribute_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['filter'][concept] = scores.detach().cpu().numpy()
                else:
                    output_list[i]['filter'][concept] = (scores > 0).nonzero().squeeze(-1).cpu().tolist()

            output_list[i]['relate_ae'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                cross_scores = self.attribute_taxnomy.cross_similarity(f, attr)
                if _apply_self_mask['relate_ae']:
                    cross_scores = do_apply_self_mask(cross_scores)
                if self.quasi:
                    output_list[i]['relate_ae'][attr] = cross_scores.detach().cpu().numpy()
                else:
                    cross_scores = cross_scores > 0
                    output_list[i]['relate_ae'][attr] = cross_scores.nonzero().cpu().tolist()

            output_list[i]['query'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                scores, word2idx = self.attribute_taxnomy.query_attribute(f, attr)
                idx2word = {v: k for k, v in word2idx.items()}
                if self.quasi:
                    output_list[i]['query'][attr] = scores.detach().cpu().numpy(), idx2word
                else:
                    argmax = scores.argmax(-1)
                    output_list[i]['query'][attr] = [idx2word[v] for v in argmax.cpu().tolist()]

            f = f_sng[i][2]

            output_list[i]['relate'] = dict()
            for concept in self.relation_taxnomy.all_concepts:
                scores = self.relation_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['relate'][concept] = scores.detach().cpu().numpy()
                else:
                    output_list[i]['relate'][concept] = (scores > 0).nonzero().cpu().tolist()

            output_list[i]['nr_objects'] = nr_objects

        return output_list


class ProgramExecutorContext(nn.Module):
    def __init__(self, attribute_taxnomy, relation_taxnomy, features, parameter_resolution, training=True):
        super().__init__()

        self.features = features
        self.parameter_resolution = ParameterResolutionMode.from_string(parameter_resolution)

        # None, attributes, relations
        self.taxnomy = [None, attribute_taxnomy, relation_taxnomy]
        self._concept_groups_masks = [None, None, None]

        self._attribute_groups_masks = None
        self._attribute_query_masks = None
        self._attribute_query_ls_masks = None
        self._attribute_query_ls_mc_masks = None

        self.train(training)

    def filter(self, selected, group, concept_groups):
        if group is None:
            return selected
        mask = self._get_concept_groups_masks(concept_groups, 1)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_most(self, selected, group, concept_groups):
        mask = self._get_concept_groups_masks(concept_groups, 2)

        # mask[x] = \exists y \in selected, greater(y, x)
        mask = torch.min(mask, selected.unsqueeze(-1).unsqueeze(0)).max(dim=-2)[0]
        # -mask[x] = \forall y \in selected, less_eq(y, x)
        mask = torch.min(selected, -mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def relate(self, selected, group, concept_groups):
        mask = self._get_concept_groups_masks(concept_groups, 2)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def relate_ae(self, selected, group, attribute_groups):
        mask = self._get_attribute_groups_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def unique(self, selected):
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            return jacf.general_softmax(selected, impl='standard', training=self.training)
        # trigger the greedy_max
        return jacf.general_softmax(selected, impl='gumbel_hard', training=self.training)

    def intersect(self, selected1, selected2):
        return torch.min(selected1, selected2)

    def union(self, selected1, selected2):
        return torch.max(selected1, selected2)

    def exist(self, selected):
        return selected.max(dim=-1)[0]

    def belong_to(self, selected1, selected2):
        return (selected1 * selected2).sum(dim=-1)

    def count(self, selected):
        if self.training:
            return torch.sigmoid(selected).sum(dim=-1)
        else:
            if _test_quantize.value >= InferenceQuantizationMethod.STANDARD.value:
                return (selected > 0).float().sum()
            return torch.sigmoid(selected).sum(dim=-1).round()

    _count_margin = 0.25
    _count_tau = 0.25

    def count_greater(self, selected1, selected2):
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            a = torch.sigmoid(selected1).sum(dim=-1)
            b = torch.sigmoid(selected2).sum(dim=-1)

            return ((a - b - 1 + 2 * self._count_margin) / self._count_tau)
        else:
            return -10 + 20 * (self.count(selected1) > self.count(selected2)).float()

    def count_less(self, selected1, selected2):
        return self.count_greater(selected2, selected1)

    def count_equal(self, selected1, selected2):
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            a = torch.sigmoid(selected1).sum(dim=-1)
            b = torch.sigmoid(selected2).sum(dim=-1)
            return ((2 * self._count_margin - (a - b).abs()) / (2 * self._count_margin) / self._count_tau)
        else:
            return -10 + 20 * (self.count(selected1) == self.count(selected2)).float()

    def query(self, selected, group, attribute_groups):
        mask, word2idx = self._get_attribute_query_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx

    def query_ls(self, selected, group, attribute_groups):
        """large-scale query"""
        mask, word2idx = self._get_attribute_query_ls_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx

    def query_ls_mc(self, selected, group, attribute_groups, concepts):
        mask, word2idx = self._get_attribute_query_ls_mc_masks(attribute_groups, concepts)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx

    def query_is(self, selected, group, concept_groups):
        mask = self._get_concept_groups_masks(concept_groups, 1)
        mask = (mask * selected.unsqueeze(0)).sum(dim=-1)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def query_ae(self, selected1, selected2, group, attribute_groups):
        mask = self._get_attribute_groups_masks(attribute_groups)
        mask = (mask * selected1.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        mask = (mask * selected2.unsqueeze(0)).sum(dim=-1)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def _get_concept_groups_masks(self, concept_groups, k):
        if self._concept_groups_masks[k] is None:
            masks = list()
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    new_mask = self.taxnomy[k].similarity(self.features[k], c)
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if k == 2 and _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_attribute_groups_masks(self, attribute_groups):
        if self._attribute_groups_masks is None:
            masks = list()
            for attribute in attribute_groups:
                mask = self.taxnomy[1].cross_similarity(self.features[1], attribute)
                if _apply_self_mask['relate_ae']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._attribute_groups_masks = torch.stack(masks, dim=0)
        return self._attribute_groups_masks

    def _get_attribute_query_masks(self, attribute_groups):
        if self._attribute_query_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                # sanity check.
                if word2idx is not None:
                    for k in word2idx:
                        assert word2idx[k] == this_word2idx[k]
                word2idx = this_word2idx

            self._attribute_query_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_masks

    def _get_attribute_query_ls_masks(self, attribute_groups):
        if self._attribute_query_ls_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                word2idx = this_word2idx

            self._attribute_query_ls_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_ls_masks

    def _get_attribute_query_ls_mc_masks(self, attribute_groups, concepts):
        if self._attribute_query_ls_mc_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute_mc(self.features[1], attribute, concepts)
                masks.append(mask)
                word2idx = this_word2idx

            self._attribute_query_ls_mc_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_ls_mc_masks


class DifferentiableReasoning(nn.Module):
    def __init__(self, used_concepts, input_dims, hidden_dims, parameter_resolution='deterministic', vse_attribute_agnostic=False):
        super().__init__()

        self.used_concepts = used_concepts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.parameter_resolution = parameter_resolution

        for i, nr_vars in enumerate(['attribute', 'relation']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars, concept_embedding.ConceptEmbedding(vse_attribute_agnostic))
            tax = getattr(self, 'embedding_' + nr_vars)
            rec = self.used_concepts[nr_vars]

            for a in rec['attributes']:
                tax.init_attribute(a, self.input_dims[1 + i], self.hidden_dims[1 + i])
            for (v, b) in rec['concepts']:
                tax.init_concept(v, self.hidden_dims[1 + i], known_belong=b)

        for i, nr_vars in enumerate(['attribute_ls', 'relation_ls']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars.replace('_ls', ''), concept_embedding_ls.ConceptEmbeddingLS(
                self.input_dims[1 + i], self.hidden_dims[1 + i], self.hidden_dims[1 + i]
            ))
            tax = getattr(self, 'embedding_' + nr_vars.replace('_ls', ''))
            rec = self.used_concepts[nr_vars]

            if rec['attributes'] is not None:
                tax.init_attributes(rec['attributes'], self.used_concepts['embeddings'])
            if rec['concepts'] is not None:
                tax.init_concepts(rec['concepts'], self.used_concepts['embeddings'])

    def forward(self, batch_features, progs, fd=None):
        assert len(progs) == len(batch_features)

        programs = []
        buffers = []
        result = []
        for i, (features, prog) in enumerate(zip(batch_features, progs)):
            buffer = []

            buffers.append(buffer)
            programs.append(prog)

            ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, features, parameter_resolution=self.parameter_resolution, training=self.training)

            for block_id, block in enumerate(prog):
                op = block['op']

                if op == 'scene':
                    buffer.append(10 + torch.zeros(features[1].size(0), dtype=torch.float, device=features[1].device))
                    continue

                inputs = []
                for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                    inp = buffer[inp]
                    if inp_type == 'object':
                        inp = ctx.unique(inp)
                    inputs.append(inp)

                # TODO(Jiayuan Mao @ 10/06): add support of soft concept attention.

                if op == 'filter':
                    buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                elif op == 'filter_scene':
                    inputs = [10 + torch.zeros(features[1].size(0), dtype=torch.float, device=features[1].device)]
                    buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                elif op == 'filter_most':
                    buffer.append(ctx.filter_most(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                elif op == 'relate':
                    buffer.append(ctx.relate(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                elif op == 'relate_attribute_equal':
                    buffer.append(ctx.relate_ae(*inputs, block['attribute_idx'], block['attribute_values']))
                elif op == 'intersect':
                    buffer.append(ctx.intersect(*inputs))
                elif op == 'union':
                    buffer.append(ctx.union(*inputs))
                else:
                    assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                    if op == 'query':
                        buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                    elif op == 'query_ls':
                        buffer.append(ctx.query_ls(*inputs, block['attribute_idx'], block['attribute_values']))
                    elif op == 'query_ls_mc':
                        buffer.append(ctx.query_ls_mc(*inputs, block['attribute_idx'], block['attribute_values'], block['multiple_choices']))
                    elif op == 'query_is':
                        buffer.append(ctx.query_is(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'query_attribute_equal':
                        buffer.append(ctx.query_ae(*inputs, block['attribute_idx'], block['attribute_values']))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'belong_to':
                        buffer.append(ctx.belong_to(*inputs))
                    elif op == 'count':
                        buffer.append(ctx.count(*inputs))
                    elif op == 'count_greater':
                        buffer.append(ctx.count_greater(*inputs))
                    elif op == 'count_less':
                        buffer.append(ctx.count_less(*inputs))
                    elif op == 'count_equal':
                        buffer.append(ctx.count_equal(*inputs))
                    else:
                        raise NotImplementedError('Unsupported operation: {}.'.format(op))

                if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                    if block_id != len(prog) - 1:
                        buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()

            result.append((op, buffer[-1]))

            quasi_symbolic_debug.embed(self, i, buffer, result, fd)

        return programs, buffers, result
