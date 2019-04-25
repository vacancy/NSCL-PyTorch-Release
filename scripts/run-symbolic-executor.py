#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run-symbolic-executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
A script tests the symbolic executor based on the ground-truth scene annotation and program parsing.
"""

import os.path as osp

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jacinle.utils.tqdm import tqdm_gofor, get_current_tqdm
from nscl.datasets import get_available_symbolic_datasets, initialize_dataset, get_symbolic_dataset_builder
from nscl.datasets.common.program_executor import execute_program

logger = get_logger(__file__)

parser = JacArgumentParser()
parser.add_argument('--dataset', required=True, choices=get_available_symbolic_datasets(), help='dataset')
parser.add_argument('--data-dir', required=True)
parser.add_argument('--data-scenes-json', type='checked_file')
parser.add_argument('--data-questions-json', type='checked_file')
parser.add_argument('--data-vocab-json', type='checked_file')
args = parser.parse_args()

if args.data_scenes_json is None:
    args.data_scenes_json = osp.join(args.data_dir, 'scenes.json')
if args.data_questions_json is None:
    args.data_questions_json = osp.join(args.data_dir, 'questions.json')
if args.data_vocab_json is None:
    args.data_vocab_json = osp.join(args.data_dir, 'vocab.json')


def main():
    initialize_dataset(args.dataset)
    build_symbolic_dataset = get_symbolic_dataset_builder(args.dataset)
    dataset = build_symbolic_dataset(args)
    dataloader = dataset.make_dataloader(32, False, False, nr_workers=4)
    meters = GroupMeters()

    for idx, feed_dict in tqdm_gofor(dataloader):
        feed_dict = GView(feed_dict)

        for i, (p, s, gt) in enumerate(zip(feed_dict.program_seq, feed_dict.scene, feed_dict.answer)):
            _, pred = execute_program(p, s)

            if pred[0] == 'error':
                raise pred[1]

            if pred[1] != gt:
                print(p)
                print(s)

                from IPython import embed; embed()
                from sys import exit; exit()

            meters.update('accuracy', pred[1] == gt)
        get_current_tqdm().set_description(meters.format_simple('Exec:', 'val', compressed=True))

    logger.critical(meters.format_simple('Symbolic execution test:', 'avg', compressed=False))


if __name__ == '__main__':
    main()

