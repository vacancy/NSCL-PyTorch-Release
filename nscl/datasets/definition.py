#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : definition.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from jacinle.utils.cache import cached_property
from nscl.datasets.common.scene_annotation import annotate_objects

__all__ = ['DatasetDefinitionBase', 'get_global_definition', 'set_global_definition', 'gdef']


class DatasetDefinitionBase(object):
    parameter_types = ['concept', 'relational_concept', 'attribute']
    variable_types = ['object', 'object_set']
    return_types = ['word', 'integer', 'bool']

    extra_embeddings = list()

    operation_signatures = dict()

    @cached_property
    def operation_signatures_dict(self):
        return {v[0]: v[1:] for v in self.operation_signatures}

    # Automatically generated type mappings.
    @cached_property
    def qtype2atype(self):
        return [
            (name, ret_type) for name, _, _, ret_type in self.operation_signatures  # if ret_type in self.return_types
        ]

    @cached_property
    def qtype2atype_dict(self):
        return dict(self.qtype2atype)

    @cached_property
    def atype2qtypes(self):
        atype2qtypes = dict()
        for k, v in self.qtype2atype:
            atype2qtypes.setdefault(k, []).append(v)
        return atype2qtypes

    attribute_concepts = dict()

    @cached_property
    def all_attributes(self):
        return list(self.attribute_concepts.keys())

    @cached_property
    def all_attribute_concepts(self):
        return {v for vs in self.attribute_concepts.values() for v in vs}

    relational_concepts = dict()

    @cached_property
    def all_relational_concepts(self):
        return {v for vs in self.relational_concepts.values() for v in vs}

    @cached_property
    def all_concepts(self):
        return {
            'attribute': self.attribute_concepts,
            'relation': self.relational_concepts
        }

    @cached_property
    def concept2attribute(self):
        concept2attribute = dict()
        concept2attribute.update({
            v: k for k, vs in self.attribute_concepts.items() for v in vs
        })
        concept2attribute.update({
            v: k for k, vs in self.relational_concepts.items() for v in vs
        })
        return concept2attribute

    def translate_scene(self, scene):
        return scene

    def translate_question(self, question):
        return question

    def get_image_filename(self, scene):
        return scene['image_filename']

    def annotate_scene(self, scene):
        raise NotImplementedError()

    def annotate_objects(self, scene):
        return annotate_objects(scene)

    def annotate_question_metainfo(self, metainfo):
        raise NotImplementedError()

    def annotate_question(self, metainfo):
        raise NotImplementedError()

    def program_to_nsclseq(self, program, question=None):
        raise NotImplementedError()

    def canonize_answer(self, answer, question_type):
        raise NotImplementedError()

    def update_collate_guide(self, collate_guide):
        raise NotImplementedError()


class GlobalDefinitionWrapper(object):
    def __getattr__(self, item):
        return getattr(get_global_definition(), item)

    def __setattr__(self, key, value):
        raise AttributeError('Cannot set the attr of `gdef`.')


gdef = GlobalDefinitionWrapper()


_GLOBAL_DEF = None


def get_global_definition():
    global _GLOBAL_DEF
    assert _GLOBAL_DEF is not None
    return _GLOBAL_DEF


def set_global_definition(def_):
    global _GLOBAL_DEF
    assert _GLOBAL_DEF is None
    _GLOBAL_DEF = def_
