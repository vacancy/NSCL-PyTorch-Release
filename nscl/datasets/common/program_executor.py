#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/29/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Program executor on the CLEVR dataset.
"""

import six
from copy import deepcopy
import numpy as np
from scipy.special import expit

from jacinle.utils.enum import JacEnum
from nscl.datasets.definition import gdef

__all__ = [
    'ParameterResolutionMode',
    'InvalidObjectReference', 'AmbiguousObjectReference',
    'ProgramExecutor', 'ConceptQuantizationProgramExecutor',
    'execute_program', 'execute_program_concept_quantization', 'execute_program_quasi_concept_quantization'
]


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class ParameterResolutionMode(JacEnum):
    DETERMINISTIC = 'deterministic'
    PROBABILISTIC_SAMPLE = 'probabilistic_sample'
    PROBABILISTIC_ARGMAX = 'probabilistic_argmax'


class InvalidObjectReference(Exception):
    pass


class AmbiguousObjectReference(Exception):
    pass


class ProgramExecutor(object):
    def __init__(self, parameter_resolution='deterministic'):
        super().__init__()
        self.parameter_resolution = ParameterResolutionMode.from_string(parameter_resolution)

    def execute(self, program, scene, reference_sanity_check=True):
        buffer = []
        result = (None, None)

        try:
            for block in program:
                op = block['op']

                if op == 'scene':
                    buffer.append(self.scene(scene))
                    continue

                inputs = [buffer[i] for i in block['inputs']]

                if op == 'filter':
                    buffer.append(self.filter(scene, *inputs, self._resolve_parameter(block['concept'])))
                elif op == 'filter_most':
                    buffer.append(self.filter_most(scene, *inputs, self._resolve_parameter(block['most_concept'])))
                elif op == 'relate':
                    self._check_unique(inputs[0], reference_sanity_check)
                    buffer.append(self.relate(scene, *inputs, self._resolve_parameter(block['relational_concept'])))
                elif op == 'relate_attribute_equal':
                    self._check_unique(inputs[0], reference_sanity_check)
                    buffer.append(self.relate_ae(scene, *inputs, self._resolve_parameter(block['attribute'])))
                elif op == 'intersect':
                    buffer.append(np.minimum(*inputs))
                elif op == 'union':
                    buffer.append(np.maximum(*inputs))
                else:
                    if op == 'query':
                        self._check_unique(inputs[0], reference_sanity_check)
                        buffer.append(str(self.query(scene, *inputs, self._resolve_parameter(block['attribute']))))
                    elif op == 'query_is':
                        self._check_unique(inputs[0], reference_sanity_check)
                        buffer.append(bool(self.query_is(scene, *inputs, self._resolve_parameter(block['concept']))))
                    elif op == 'query_attribute_equal':
                        self._check_unique(inputs[0], reference_sanity_check)
                        self._check_unique(inputs[1], reference_sanity_check)
                        buffer.append(bool(self.query_ae(scene, *inputs, self._resolve_parameter(block['attribute']))))
                    elif op == 'exist':
                        buffer.append(bool(inputs[0].max() > 0))
                    elif op == 'count':
                        buffer.append(int(self.count(inputs[0])))
                    elif op == 'count_greater':
                        buffer.append(bool(self.count(inputs[0]) > self.count(inputs[1])))
                    elif op == 'count_less':
                        buffer.append(bool(self.count(inputs[0]) < self.count(inputs[1])))
                    elif op == 'count_equal':
                        buffer.append(bool(self.count(inputs[0]) == self.count(inputs[1])))
                    elif op == 'belong_to':
                        self._check_unique(inputs[0], reference_sanity_check)
                        buffer.append(bool(self.belong_to(inputs[0], inputs[1])))
                    else:
                        raise NotImplementedError('Unsupported operation: {}.'.format(op))
                    result = (op, buffer[-1])

        except (InvalidObjectReference, AmbiguousObjectReference) as e:
            return [], ('error', e)
        except Exception as e:
            raise e

        return buffer, result

    def _resolve_parameter(self, parameter):
        if self.parameter_resolution is ParameterResolutionMode.DETERMINISTIC:
            return parameter
        raise NotImplementedError('Unimplemented parameter resolution: {}.'.format(self.parameter_resolution))

    def _check_unique(self, x, activate):
        if activate:
            if x.sum() < 1:
                raise InvalidObjectReference()
            elif x.sum() > 1:
                raise AmbiguousObjectReference()

    def scene(self, scene):
        return np.ones(len(scene['objects']), dtype='float32')

    def filter(self, scene, x, filters):
        objects = scene['objects']
        y = np.ones_like(x)
        for i, o in enumerate(objects):
            for f in filters:
                attr = gdef.concept2attribute[f]
                if (isinstance(o[attr], six.string_types) and o[attr] != f) or (isinstance(o[attr], (tuple, list)) and f not in o[attr]):
                    y[i] = 0
                    break
        return np.minimum(x, y)

    def filter_most(self, scene, x, concept):
        relations = scene['relationships']
        assert len(concept) == 1
        concept = concept[0]

        y = x.copy()
        for i, er_list in enumerate(relations[concept]):
            if x[i] == 0:
                continue
            for j in er_list:
                if x[j] == 1:
                    y[i] = 0
                    break
        return y

    def relate(self, scene, x, f):
        relations = scene['relationships']
        t = x.argmax(-1)
        assert len(f) == 1
        f = f[0]
        y = np.ones_like(x)
        for i in range(len(y)):
            if i not in relations[f][t]:
                y[i] = 0
        return y

    def relate_ae(self, scene, x, attr):
        objects = scene['objects']
        t = x.argmax(-1)
        y = np.ones_like(x)
        for i, o in enumerate(objects):
            if o[attr] != objects[t][attr] or i == t:
                y[i] = 0
        return y

    def query(self, scene, x, attr):
        objects = scene['objects']
        t = x.argmax(-1)
        return objects[t][attr]

    def query_is(self, scene, x, concept):
        objects = scene['objects']
        assert len(concept) == 1
        concept = concept[0]

        t = x.argmax(-1)
        attr = gdef.concept2attribute[concept]
        return (isinstance(objects[t][attr], six.string_types) and objects[t][attr] == concept) or \
                (isinstance(objects[t][attr], (tuple, list)) and concpet in objects[t][attr])

    def query_ae(self, scene, x, y, attr):
        objects = scene['objects']
        u = x.argmax(-1)
        v = y.argmax(-1)

        return objects[u][attr] == objects[v][attr]

    def count(self, x):
        return int(x.sum())

    def belong_to(self, x, y):
        return (x * y).max() > 0


class ConceptQuantizationProgramExecutor(ProgramExecutor):
    def __init__(self, quasi=False, parameter_resolution='deterministic'):
        super().__init__(parameter_resolution=parameter_resolution)
        self.quasi = quasi

    @staticmethod
    def process_scene(scene, copy=True):
        # NB(Jiayuan Mao @ 04/15): we actually don't need a copy.
        if copy:
            scene = deepcopy(scene)

        for k in scene['filter']:
            scene['filter'][k] = scene['filter'][k]
        for k in scene['relate']:
            scene['relate'][k] = {tuple(v) for v in scene['relate'][k]}
        for k in scene['relate_ae']:
            scene['relate_ae'][k] = {tuple(v) for v in scene['relate_ae'][k]}
        for k in scene['query']:
            scene['query'][k] = scene['query'][k]

        return scene


    @staticmethod
    def _unique_softmax(mask):
        return softmax(mask, axis=-1)

    def scene(self, scene):
        return np.ones(scene['nr_objects'], dtype='float32')

    def filter(self, scene, x, filters):
        y = np.ones_like(x)
        for f in filters:
            if self.quasi:
                z = scene['filter'][f]
            else:
                z = np.zeros_like(y)
                z[scene['filter'][f]] = 1
            y = np.minimum(y, z)
        return np.minimum(x, y)

    def filter_most(self, scene, x, concept):
        raise NotImplementedError()

    def relate(self, scene, x, f):
        assert len(f) == 1
        f = f[0]

        if self.quasi:
            x = self._unique_softmax(x)
            return np.dot(x, scene['relate'][f])
        else:
            t = x.argmax(-1)
            y = np.ones_like(x)
            for i in range(len(y)):
                if (t, i) not in scene['relate'][f]:
                    y[i] = 0
            return y

    def relate_ae(self, scene, x, attr):
        if self.quasi:
            x = self._unique_softmax(x)
            return np.dot(x, scene['relate_ae'][attr])
        else:
            t = x.argmax(-1)
            y = np.ones_like(x)
            for i in range(len(y)):
                if (t, i) not in scene['relate_ae'][attr] or i == t:
                    y[i] = 0
            return y

    def query(self, scene, x, attr):
        if self.quasi:
            x = self._unique_softmax(x)
            W, idx2word = scene['query'][attr]
            return idx2word[np.dot(x, W).argmax()]
        else:
            t = x.argmax(-1)
            return scene['query'][attr][t]

    def query_is(self, scene, x, concept):
        raise NotImplementedError()

    def query_ae(self, scene, x, y, attr):
        if self.quasi:
            x = self._unique_softmax(x)
            y = self._unique_softmax(y)
            return np.dot(np.dot(x, scene['relate_ae'][attr]), y)
        else:
            u = x.argmax(-1)
            v = y.argmax(-1)
            return (u, v) in scene['relate_ae'][attr]

    def count(self, x):
        if self.quasi:
            return np.round(expit(x).sum())
        else:
            return x.sum()

    def belong_to(self, x, y):
        raise NotImplementedError()


_exe = ProgramExecutor()


def execute_program(program, scene):
    return _exe.execute(program, scene)


_cqexe = ConceptQuantizationProgramExecutor()


def execute_program_concept_quantization(program, scene):
    return _cqexe.execute(program, scene, reference_sanity_check=False)


_cqexe2 = ConceptQuantizationProgramExecutor(quasi=True)


def execute_program_quasi_concept_quantization(program, scene):
    return _cqexe2.execute(progarm, scene, reference_sanity_check=False)

