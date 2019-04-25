#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_analysis.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/30/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Program analytics tools.
"""

import collections

from nscl.datasets.definition import gdef

__all__ = ['dfs_nscltree', 'nscltree_get_depth', 'nscltree_stat_parameters', 'nscltree_to_string', 'nscltree_to_string_full']


def dfs_nscltree(program):
    def dfs(pblock):
        yield pblock
        for i in pblock['inputs']:
            yield from dfs(i)

    return list(dfs(program))


def nscltree_get_depth(program):
    def dfs(pblock):
        if 'inputs' not in pblock or len(pblock['inputs']) == 0:
            return 1
        return 1 + max(dfs(p) for p in pblock['inputs'])
    try:
        return dfs(program)
    finally:
        del dfs


def nscltree_stat_parameters(program):
    result = collections.defaultdict(int)
    for pblock in dfs_nscltree(program):
        op = pblock['op']
        for x in gdef.operation_signatures_dict[op][0]:
            if x in gdef.parameter_types:
                result[x] += 1
    return result


def nscltree_to_string(program):
    def dfs(pblock):
        ret = pblock['op'] + '('
        inputs = [dfs(i) for i in pblock['inputs']]
        ret += ','.join(inputs)
        ret += ')'
        return ret
    return dfs(program)


def nscltree_to_string_full(program):
    def dfs(pblock):
        ret = pblock['op'] + '('
        inputs = []
        for param_type in gdef.parameter_types:
            param_record = None
            if param_type in pblock:
                param_record = pblock[param_type]
            elif param_type + '_idx' in pblock:
                param_record = pblock[param_type + '_values'][pblock[param_type + '_idx']]

            if param_record is not None:
                param_str = '|'.join(param_record) if isinstance(param_record, (tuple, list)) else str(param_record)
                inputs.append(param_str)

        inputs.extend([dfs(i) for i in pblock['inputs']])
        ret += ','.join(inputs)
        ret += ')'
        return ret

    return dfs(program)

