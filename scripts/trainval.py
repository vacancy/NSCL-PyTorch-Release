#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : trainval.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/05/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Training and evaulating the Neuro-Symbolic Concept Learner.
"""

import time
import os.path as osp

import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.train import TrainerEnv
from jactorch.utils.meta import as_float

from nscl.datasets import get_available_datasets, initialize_dataset, get_dataset_builder

logger = get_logger(__file__)

parser = JacArgumentParser(description=__doc__.strip())

parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--configs', default='', type='kv', metavar='CFGS')

# training_target and curriculum learning
parser.add_argument('--expr', default=None, metavar='DIR', help='experiment name')
parser.add_argument('--training-target', required=True, choices=['derender', 'parser', 'all'])
parser.add_argument('--training-visual-modules', default='all', choices=['none', 'object', 'relation', 'all'])
parser.add_argument('--curriculum', default='all', choices=['off', 'scene', 'program', 'all'])
parser.add_argument('--question-transform', default='off', choices=['off', 'basic', 'parserv1-groundtruth', 'parserv1-candidates', 'parserv1-candidates-executed'])
parser.add_argument('--concept-quantization-json', default=None, metavar='FILE')

# running mode
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--evaluate', action='store_true', help='run the validation only; used with --resume')

# training hyperparameters
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of total epochs to run')
parser.add_argument('--enums-per-epoch', type=int, default=1, metavar='N', help='number of enumerations of the whole dataset per epoch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='initial learning rate')
parser.add_argument('--iters-per-epoch', type=int, default=0, metavar='N', help='number of iterations per epoch 0=one pass of the dataset (default: 0)')
parser.add_argument('--acc-grad', type=int, default=1, metavar='N', help='accumulated gradient (default: 1)')
parser.add_argument('--clip-grad', type=float, metavar='F', help='gradient clipping')
parser.add_argument('--validation-interval', type=int, default=1, metavar='N', help='validation inverval (epochs) (default: 1)')

# finetuning and snapshot
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')
parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='manual epoch number')
parser.add_argument('--save-interval', type=int, default=2, metavar='N', help='model save interval (epochs) (default: 10)')

# data related
parser.add_argument('--dataset', required=True, choices=get_available_datasets(), help='dataset')
parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--data-trim', type=float, default=0, metavar='F', help='trim the dataset')
parser.add_argument('--data-split',type=float, default=0.75, metavar='F', help='fraction / numer of training samples')
parser.add_argument('--data-vocab-json', type='checked_file', metavar='FILE')
parser.add_argument('--data-scenes-json', type='checked_file', metavar='FILE')
parser.add_argument('--data-questions-json', type='checked_file', metavar='FILE', nargs='+')

parser.add_argument('--extra-data-dir', type='checked_dir', metavar='DIR', help='extra data directory for validation')
parser.add_argument('--extra-data-scenes-json', type='checked_file', nargs='+', default=None, metavar='FILE', help='extra scene json file for validation')
parser.add_argument('--extra-data-questions-json', type='checked_file', nargs='+', default=None, metavar='FILE', help='extra question json file for validation')

parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', type='bool', default=False, metavar='B', help='use tensorboard or not')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()

if args.data_vocab_json is None:
    args.data_vocab_json = osp.join(args.data_dir, 'vocab.json')

args.data_image_root = osp.join(args.data_dir, 'images')
if args.data_scenes_json is None:
    args.data_scenes_json = osp.join(args.data_dir, 'scenes.json')
if args.data_questions_json is None:
    args.data_questions_json = osp.join(args.data_dir, 'questions.json')

if args.extra_data_dir is not None:
    args.extra_data_image_root = osp.join(args.extra_data_dir, 'images')
    if args.extra_data_scenes_json is None:
        args.extra_data_scenes_json = osp.join(args.extra_data_dir, 'scenes.json')
    if args.extra_data_questions_json is None:
        args.extra_data_questions_json = osp.join(args.extra_data_dir, 'questions.json')

# filenames
args.series_name = args.dataset
args.desc_name = escape_desc_name(args.desc)
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

# directories

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)

desc = load_source(args.desc)
configs = desc.configs
args.configs.apply(configs)


def main():
    args.dump_dir = ensure_path(osp.join(
        'dumps', args.series_name, args.desc_name, (
            args.training_target +
            ('-curriculum_' + args.curriculum) +
            ('-qtrans_' + args.question_transform if args.question_transform is not None else '') +
            ('-' + args.expr if args.expr is not None else '')
        )
    ))

    if not args.debug:
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.meta_dir = ensure_path(osp.join(args.dump_dir, 'meta'))
        args.meta_file = osp.join(args.meta_dir, args.run_name + '.json')
        args.log_file = osp.join(args.meta_dir, args.run_name + '.log')
        args.meter_file = osp.join(args.meta_dir, args.run_name + '.meter.json')

        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir_root = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
            args.tb_dir = ensure_path(osp.join(args.tb_dir_root, args.run_name))

    initialize_dataset(args.dataset)
    build_dataset = get_dataset_builder(args.dataset)

    dataset = build_dataset(args, configs, args.data_image_root, args.data_scenes_json, args.data_questions_json)

    dataset_trim = int(len(dataset) * args.data_trim) if args.data_trim <= 1 else int(args.data_trim)
    if dataset_trim > 0:
        dataset = dataset.trim_length(dataset_trim)

    dataset_split = int(len(dataset) * args.data_split) if args.data_split <= 1 else int(args.data_split)
    train_dataset, validation_dataset = dataset.split_trainval(dataset_split)

    extra_dataset = None
    if args.extra_data_dir is not None:
        extra_dataset = build_dataset(args, configs, args.extra_data_image_root, args.extra_data_scenes_json, args.extra_data_questions_json)

    main_train(train_dataset, validation_dataset, extra_dataset)


def main_train(train_dataset, validation_dataset, extra_dataset=None):
    logger.critical('Building the model.')
    model = desc.make_model(args, train_dataset.unwrapped.vocab)

    if args.use_gpu:
        model.cuda()
        # Use the customized data parallel if applicable.
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            # from jactorch.parallel import UserScatteredJacDataParallel as JacDataParallel
            model = JacDataParallel(model, device_ids=args.gpus).cuda()
        # Disable the cudnn benchmark.
        cudnn.benchmark = False

    if hasattr(desc, 'make_optimizer'):
        logger.critical('Building customized optimizer.')
        optimizer = desc.make_optimizer(model, args.lr)
    else:
        from jactorch.optim import AdamW
        trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = AdamW(trainable_parameters, args.lr, weight_decay=configs.train.weight_decay)

    if args.acc_grad > 1:
        from jactorch.optim import AccumGrad
        optimizer = AccumGrad(optimizer, args.acc_grad)
        logger.warning('Use accumulated grad={:d}, effective iterations per epoch={:d}.'.format(args.acc_grad, int(args.iters_per_epoch / args.acc_grad)))

    trainer = TrainerEnv(model, optimizer)

    if args.resume:
        extra = trainer.load_checkpoint(args.resume)
        if extra:
            args.start_epoch = extra['epoch']
            logger.critical('Resume from epoch {}.'.format(args.start_epoch))
    elif args.load:
        if trainer.load_weights(args.load):
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))

    if args.use_tb and not args.debug:
        from jactorch.train.tb import TBLogger, TBGroupMeters
        tb_logger = TBLogger(args.tb_dir)
        meters = TBGroupMeters(tb_logger)
        logger.critical('Writing tensorboard logs to: "{}".'.format(args.tb_dir))
    else:
        from jacinle.utils.meter import GroupMeters
        meters = GroupMeters()

    if not args.debug:
        logger.critical('Writing meter logs to file: "{}".'.format(args.meter_file))

    if args.clip_grad:
        logger.info('Registering the clip_grad hook: {}.'.format(args.clip_grad))
        def clip_grad(self, loss):
            from torch.nn.utils import clip_grad_norm_
            clip_grad_norm_(self.model.parameters(), max_norm=args.clip_grad)
        trainer.register_event('backward:after', clip_grad)

    if hasattr(desc, 'customize_trainer'):
        desc.customize_trainer(trainer)

    if args.embed:
        from IPython import embed; embed()

    logger.critical('Building the data loader.')
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)
    if extra_dataset is not None:
        extra_dataloader = extra_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    if args.evaluate:
        meters.reset()
        model.eval()
        validate_epoch(0, trainer, validation_dataloader, meters)
        if extra_dataset is not None:
            validate_epoch(0, trainer, extra_dataloader, meters, meter_prefix='validation_extra')
        logger.critical(meters.format_simple('Validation', {k: v for k, v in meters.avg.items() if v != 0}, compressed=False))
        return meters

    # assert args.curriculum == 'off', 'Unimplemented feature: curriculum mode {}.'.format(args.curriculum)
    curriculum_strategy = [
        (0, 3, 4),
        (5, 3, 6),
        (10, 3, 8),
        (15, 4, 8),
        (25, 4, 12),
        (35, 5, 12),
        (45, 6, 12),
        (55, 7, 16),
        (65, 8, 20),
        (75, 9, 22),
        (90, 10, 25),
        (1e9, None, None)
    ]

    # trainer.register_event('backward:after', backward_check_nan)

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        meters.reset()

        model.train()

        this_train_dataset = train_dataset
        if args.curriculum != 'off':
            for si, s in enumerate(curriculum_strategy):
                if curriculum_strategy[si][0] < epoch <= curriculum_strategy[si + 1][0]:
                    max_scene_size, max_program_size = s[1:]
                    if args.curriculum in ('scene', 'all'):
                        this_train_dataset = this_train_dataset.filter_scene_size(max_scene_size)
                    if args.curriculum in ('program', 'all'):
                        this_train_dataset = this_train_dataset.filter_program_size_raw(max_program_size)
                    logger.critical('Building the data loader. Curriculum = {}/{}, length = {}.'.format(*s[1:], len(this_train_dataset)))
                    break

        train_dataloader = this_train_dataset.make_dataloader(args.batch_size, shuffle=True, drop_last=True, nr_workers=args.data_workers)

        for enum_id in range(args.enums_per_epoch):
            train_epoch(epoch, trainer, train_dataloader, meters)

        if epoch % args.validation_interval == 0:
            model.eval()
            validate_epoch(epoch, trainer, validation_dataloader, meters)

        if not args.debug:
            meters.dump(args.meter_file)

        logger.critical(meters.format_simple(
            'Epoch = {}'.format(epoch),
            {k: v for k, v in meters.avg.items() if epoch % args.validation_interval == 0 or not k.startswith('validation')},
            compressed=False
        ))

        if epoch % args.save_interval == 0 and not args.debug:
            fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
            trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))

        if epoch > int(args.epochs * 0.6):
            trainer.set_learning_rate(args.lr * 0.1)


def backward_check_nan(self, feed_dict, loss, monitors, output_dict):
    import torch
    for name, param in self.model.named_parameters():
        if param.grad is None:
            continue
        if torch.isnan(param.grad.data).any().item():
            print('Caught NAN in gradient.', name)
            from IPython import embed; embed()


def train_epoch(epoch, trainer, train_dataloader, meters):
    nr_iters = args.iters_per_epoch
    if nr_iters == 0:
        nr_iters = len(train_dataloader)

    meters.update(epoch=epoch)

    trainer.trigger_event('epoch:before', trainer, epoch)
    train_iter = iter(train_dataloader)

    end = time.time()
    with tqdm_pbar(total=nr_iters) as pbar:
        for i in range(nr_iters):
            feed_dict = next(train_iter)

            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            loss, monitors, output_dict, extra_info = trainer.step(feed_dict, cast_tensor=False)
            step_time = time.time() - end; end = time.time()

            n = feed_dict['image'].size(0)
            meters.update(loss=loss, n=n)
            meters.update(monitors, n=n)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {}'.format(epoch),
                {k: v for k, v in meters.val.items() if not k.startswith('validation') and k != 'epoch' and k.count('/') <= 1},
                compressed=True
            ))
            pbar.update()

            end = time.time()

    trainer.trigger_event('epoch:after', trainer, epoch)


def validate_epoch(epoch, trainer, val_dataloader, meters, meter_prefix='validation'):
    end = time.time()
    with tqdm_pbar(total=len(val_dataloader)) as pbar:
        for feed_dict in val_dataloader:
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            output_dict, extra_info = trainer.evaluate(feed_dict, cast_tensor=False)
            monitors = {meter_prefix + '/' + k: v for k, v in as_float(output_dict['monitors']).items()}
            step_time = time.time() - end; end = time.time()

            n = feed_dict['image'].size(0)
            meters.update(monitors, n=n)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {} (validation)'.format(epoch),
                {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 2},
                compressed=True
            ))
            pbar.update()

            end = time.time()


if __name__ == '__main__':
    main()

