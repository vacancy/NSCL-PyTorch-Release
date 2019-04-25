#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualize-dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/18/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import os.path as osp
from PIL import Image

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jacinle.random as random
from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.utils.tqdm import tqdm
from jaclearn.visualize.box import vis_bboxes
from jaclearn.visualize.html_table import HTMLTableVisualizer, HTMLTableColumnDesc
from nscl.datasets import get_available_symbolic_datasets, initialize_dataset, get_symbolic_dataset_builder

logger = get_logger(__file__)

parser = JacArgumentParser()
parser.add_argument('--dataset', required=True, choices=get_available_symbolic_datasets(), help='dataset')
parser.add_argument('--data-dir', required=True)
parser.add_argument('--data-scenes-json', type='checked_file')
parser.add_argument('--data-questions-json', type='checked_file')
parser.add_argument('--data-vocab-json', type='checked_file')
parser.add_argument('-n', '--nr-vis', type=int, help='number of visualized questions')
parser.add_argument('--random', type='bool', default=False, help='random choose the questions')
args = parser.parse_args()

args.data_image_root = osp.join(args.data_dir, 'images')
args.data_vis_dir = osp.join(args.data_dir, 'visualize')
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

    if args.nr_vis is None:
        args.nr_vis = min(100, len(dataset))

    if args.random:
        indices = random.choice(len(dataset), size=args.nr_vis, replace=False)
    else:
        indices = list(range(args.nr_vis))

    vis = HTMLTableVisualizer(args.data_vis_dir, 'Dataset: ' + args.dataset.upper())
    vis.begin_html()
    with vis.table('Metainfo', [
        HTMLTableColumnDesc('k', 'Key', 'text', {}),
        HTMLTableColumnDesc('v', 'Value', 'code', {})
    ]):
        for k, v in args.__dict__.items():
            vis.row(k=k, v=v)

    with vis.table('Visualize', [
        HTMLTableColumnDesc('id', 'QuestionID', 'text', {}),
        HTMLTableColumnDesc('image', 'QA', 'figure', {'width': '100%'}),
        HTMLTableColumnDesc('qa', 'QA', 'text', td_css={'width': '30%'}),
        HTMLTableColumnDesc('p', 'Program', 'code', td_css={'width': '30%'})
    ]):
        for i in tqdm(indices):
            feed_dict = GView(dataset[i])
            image_filename = osp.join(args.data_image_root, feed_dict.image_filename)
            image = Image.open(image_filename)

            if 'objects' in feed_dict:
                fig, ax = vis_bboxes(image, feed_dict.objects, 'object', add_text=False)
            else:
                fig, ax = vis_bboxes(image, [], 'object', add_text=False)
            _ = ax.set_title('object bounding box annotations')

            QA_string = """
                <p><b>Q</b>: {}</p>
                <p><b>A</b>: {}</p>
            """.format(feed_dict.question_raw, feed_dict.answer)
            P_string = '\n'.join([repr(x) for x in feed_dict.program_seq])

            vis.row(id=i, image=fig, qa=QA_string, p=P_string)
            plt.close()
    vis.end_html()

    logger.info('Happy Holiday! You can find your result at "http://monday.csail.mit.edu/xiuming' + osp.realpath(args.data_vis_dir) + '".')


if __name__ == '__main__':
    main()

