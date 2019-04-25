#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_translator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from collections import defaultdict
from copy import deepcopy

from jacinle.utils.tqdm import tqdm

from nscl.datasets.definition import gdef
from .vocab import Vocab

__all__ = ['nsclseq_to_nscltree', 'nscltree_to_nsclseq', 'nsclseq_to_nsclqsseq', 'nscltree_to_nsclqstree', 'iter_nscltree', 'gen_vocab']


def nsclseq_to_nscltree(seq_program):
    def dfs(sblock):
        tblock = deepcopy(sblock)
        input_ids = tblock.pop('inputs')
        tblock['inputs'] = [dfs(seq_program[i]) for i in input_ids]
        return tblock

    try:
        return dfs(seq_program[-1])
    finally:
        del dfs


def nscltree_to_nsclseq(tree_program):
    tree_program = deepcopy(tree_program)
    seq_program = list()

    def dfs(tblock):
        sblock = tblock.copy()
        input_blocks = sblock.pop('inputs')
        sblock['inputs'] = [dfs(b) for b in input_blocks]
        seq_program.append(sblock)
        return len(seq_program) - 1

    try:
        dfs(tree_program)
        return seq_program
    finally:
        del dfs


def nsclseq_to_nsclqsseq(seq_program):
    qs_seq = deepcopy(seq_program)
    cached = defaultdict(list)

    for sblock in qs_seq:
        for param_type in gdef.parameter_types:
            if param_type in sblock:
                sblock[param_type + '_idx'] = len(cached[param_type])
                sblock[param_type + '_values'] = cached[param_type]
                cached[param_type].append(sblock[param_type])

    return qs_seq


def nscltree_to_nsclqstree(tree_program):
    qs_tree = deepcopy(tree_program)
    cached = defaultdict(list)

    for tblock in iter_nscltree(qs_tree):
        for param_type in gdef.parameter_types:
            if param_type in tblock:
                tblock[param_type + '_idx'] = len(cached[param_type])
                tblock[param_type + '_values'] = cached[param_type]
                cached[param_type].append(tblock[param_type])

    return qs_tree


def iter_nscltree(tree_program):
    yield tree_program
    for i in tree_program['inputs']:
        yield from iter_nscltree(i)


def gen_vocab(dataset):
    all_words = set()
    for i in tqdm(len(dataset), desc='Building the vocab'):
        metainfo = dataset.get_metainfo(i)
        for w in metainfo['question_tokenized']:
            all_words.add(w)

    import jaclearn.embedding.constant as const
    vocab = Vocab()
    vocab.add(const.EBD_ALL_ZEROS)
    for w in sorted(all_words):
        vocab.add(w)
    for w in [const.EBD_UNKNOWN, const.EBD_BOS, const.EBD_EOS]:
        vocab.add(w)
    for w in gdef.extra_embeddings:
        vocab.add(w)

    return vocab
