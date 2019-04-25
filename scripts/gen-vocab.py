#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen-vocab.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/11/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import os.path as osp

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from nscl.datasets import get_available_symbolic_datasets, initialize_dataset, get_symbolic_dataset_builder

logger = get_logger(__file__)

parser = JacArgumentParser()
parser.add_argument('--dataset', required=True, choices=get_available_symbolic_datasets(), help='dataset')
parser.add_argument('--data-dir', required=True)
parser.add_argument('--data-scenes-json', type='checked_file')
parser.add_argument('--data-questions-json', type='checked_file')
parser.add_argument('--output', required=True)
args = parser.parse_args()

if args.data_scenes_json is None:
    args.data_scenes_json = osp.join(args.data_dir, 'scenes.json')
if args.data_questions_json is None:
    args.data_questions_json = osp.join(args.data_dir, 'questions.json')
args.data_vocab_json = None


def main():
    initialize_dataset(args.dataset)
    build_symbolic_dataset = get_symbolic_dataset_builder(args.dataset)
    dataset = build_symbolic_dataset(args)
    dataset.unwrapped.vocab.dump_json(args.output)
    logger.critical('Vocab json dumped at: "{}".'.format(args.output))


if __name__ == '__main__':
    main()
