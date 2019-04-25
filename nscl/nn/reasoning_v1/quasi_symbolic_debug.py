#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quasi_symbolic_debug.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/03/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
A context for debugging differentiable reasoning results.
"""

import os
import json

import numpy as np

from PIL import Image

DEBUG = os.getenv('REASONING_DEBUG', 'OFF').upper()

__all__ = ['make_debug_ctx', 'embed']


def make_debug_ctx(fd, buffer, i):
    class Context(object):
        def __init__(ctx):
            ctx.stop = False
            ctx.fd = fd
            ctx.buffer = buffer
            ctx.i = i
            ctx.program = json.loads(fd['program'][i])
            ctx.scene = json.loads(fd['scene'][i])
            ctx.raw_program = json.loads(fd['raw_program'][i])

            import visdom
            ctx.vis = visdom.Visdom(port=7002)

        def backtrace(ctx):
            print('----- backtrace ----- ')
            print('See localhost:7002 for the image.')
            ctx.vis_image()
            print(ctx.program['question'])
            for p, b, raw_p in zip(ctx.program['program'], ctx.buffer, ctx.raw_program):
                print(p)
                print('Output:', b)
                print('GT:', raw_p['_output'])

        def vis_image(ctx):
            img = Image.open(ctx.fd['image_path'][i]).convert('RGB')
            img = np.array(img).transpose((2, 0, 1))
            ctx.vis.image(img)

    return Context()


def embed(self, i, buffer, result, fd):
    if not self.training and DEBUG != 'OFF':
        p, l = result[i][1], fd['answer'][i]
        if isinstance(p, tuple):
            p, word2idx = p
            p = p.argmax(-1).item()
            idx2word = {v: k for k, v in word2idx.items()}
            p = idx2word[p]
        else:
            p, l = int((p > 0).item()), int(l)

        gogogo = False
        if p == l:
            print('Correct:', p)
            if DEBUG in ('ALL', 'CORRECT'):
                gogogo = True
        else:
            print('Wrong: ', p, l)
            if DEBUG in ('ALL', 'WRONG'):
                gogogo = True

        if gogogo:
            print('Starting the tracker.')
            ctx = make_debug_ctx(fd, buffer, i)
            from IPython import embed; embed()
            if ctx.stop:
                import sys; sys.exit()
