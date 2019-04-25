#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_translator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/30/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Tools for translating programs into different formats.
"""

from copy import deepcopy

__all__ = ['clevr_to_nsclseq']


def get_clevr_pblock_op(block):
    """
    Return the operation of a CLEVR program block.
    """
    if 'type' in block:
        return block['type']
    assert 'function' in block
    return block['function']


def get_clevr_op_attribute(op):
    return op.split('_')[1]


def clevr_to_nsclseq(clevr_program):
    nscl_program = list()
    mapping = dict()

    for block_id, block in enumerate(clevr_program):
        op = get_clevr_pblock_op(block)
        current = None
        if op == 'scene':
            current = dict(op='scene')
        elif op.startswith('filter'):
            concept = block['value_inputs'][0]
            last = nscl_program[mapping[block['inputs'][0]]]
            if last['op'] == 'filter':
                last['concept'].append(concept)
            else:
                current = dict(op='filter', concept=[concept])
        elif op.startswith('relate'):
            concept = block['value_inputs'][0]
            current = dict(op='relate', relational_concept=[concept])
        elif op.startswith('same'):
            attribute = get_clevr_op_attribute(op)
            current = dict(op='relate_attribute_equal', attribute=attribute)
        elif op in ('intersect', 'union'):
            current = dict(op=op)
        elif op == 'unique':
            pass  # We will ignore the unique operations.
        else:
            if op.startswith('query'):
                if block_id == len(clevr_program) - 1:
                    attribute = get_clevr_op_attribute(op)
                    current = dict(op='query', attribute=attribute)
            elif op.startswith('equal') and op != 'equal_integer':
                attribute = get_clevr_op_attribute(op)
                current = dict(op='query_attribute_equal', attribute=attribute)
            elif op == 'exist':
                current = dict(op='exist')
            elif op == 'count':
                if block_id == len(clevr_program) - 1:
                    current = dict(op='count')
            elif op == 'equal_integer':
                current = dict(op='count_equal')
            elif op == 'less_than':
                current = dict(op='count_less')
            elif op == 'greater_than':
                current = dict(op='count_greater')
            else:
                raise ValueError('Unknown CLEVR operation: {}.'.format(op))

        if current is None:
            assert len(block['inputs']) == 1
            mapping[block_id] = mapping[block['inputs'][0]]
        else:
            current['inputs'] = list(map(mapping.get, block['inputs']))

            if '_output' in block:
                current['output'] = deepcopy(block['_output'])

            nscl_program.append(current)
            mapping[block_id] = len(nscl_program) - 1

    return nscl_program

