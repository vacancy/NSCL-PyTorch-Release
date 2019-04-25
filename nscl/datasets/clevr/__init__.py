#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/29/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from nscl.datasets.factory import register_dataset
from .definition import CLEVRDefinition
from .definition import build_clevr_dataset, build_symbolic_clevr_dataset, \
        build_concept_retrieval_clevr_dataset, build_concept_quantization_clevr_dataset

register_dataset(
    'clevr', CLEVRDefinition,
    builder=build_clevr_dataset,
    symbolic_builder=build_symbolic_clevr_dataset,
    concept_retrieval_builder=build_concept_retrieval_clevr_dataset,
    concept_quantization_builder=build_concept_quantization_clevr_dataset
)
