#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : filterable.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/14/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import jacinle.random as random
from torch.utils.data.dataset import Dataset
from jacinle.logging import get_logger

logger = get_logger(__file__)


class FilterableDatasetUnwrapped(Dataset):
    """
    A filterable dataset. User can call various `filter_*` operations to obtain a subset of the dataset.
    """
    def __init__(self):
        super().__init__()
        self.metainfo_cache = dict()

    def get_metainfo(self, index):
        if index not in self.metainfo_cache:
            self.metainfo_cache[index] = self._get_metainfo(index)
        return self.metainfo_cache[index]

    def _get_metainfo(self, index):
        raise NotImplementedError()


class FilterableDatasetView(FilterableDatasetUnwrapped):
    def __init__(self, owner_dataset, indices=None, filter_name=None, filter_func=None):
        """
        Args:
            owner_dataset (Dataset): the original dataset.
            indices (List[int]): a list of indices that was filterred out.
            filter_name (str): human-friendly name for the filter.
            filter_func (Callable): just for tracking.
        """

        super().__init__()
        self.owner_dataset = owner_dataset
        self.indices = indices
        self._filter_name = filter_name
        self._filter_func = filter_func

    @property
    def unwrapped(self):
        if self.indices is not None:
            return self.owner_dataset.unwrapped
        return self.owner_dataset

    @property
    def filter_name(self):
        return self._filter_name if self._filter_name is not None else '<anonymous>'

    @property
    def full_filter_name(self):
        if self.indices is not None:
            return self.owner_dataset.full_filter_name + '/' + self.filter_name
        return '<original>'

    @property
    def filter_func(self):
        return self._filter_func

    def collect(self, key_func):
        return {key_func(self.get_metainfo(i)) for i in range(len(self))}

    def filter(self, filter_func, filter_name=None):
        indices = []
        for i in range(len(self)):
            metainfo = self.get_metainfo(i)
            if filter_func(metainfo):
                indices.append(i)
        if len(indices) == 0:
            raise ValueError('Filter results in an empty dataset.')
        return type(self)(self, indices, filter_name, filter_func)

    def random_trim_length(self, length):
        assert length < len(self)
        logger.info('Randomly trim the dataset: #samples = {}.'.format(length))
        indices = list(random.choice(len(self), size=length, replace=False))
        return type(self)(self, indices=indices, filter_name='randomtrim[{}]'.format(length))

    def trim_length(self, length):
        assert length < len(self)
        logger.info('Trim the dataset: #samples = {}.'.format(length))
        return type(self)(self, indices=list(range(0, length)), filter_name='trim[{}]'.format(length))

    def split_trainval(self, split):
        assert split < len(self)
        nr_train = split
        nr_val = len(self) - nr_train
        logger.info('Split the dataset: #training samples = {}, #validation samples = {}.'.format(nr_train, nr_val))
        return (
                type(self)(self, indices=list(range(0, split)), filter_name='train'),
                type(self)(self, indices=list(range(split, len(self))), filter_name='val')
        )

    def split_kfold(self, k):
        assert len(self) % k == 0
        block = len(self) // k

        for i in range(k):
            yield (
                    type(self)(self, indices=list(range(0, i * block)) + list(range((i + 1) * block, len(self))), filter_name='fold{}[train]'.format(i + 1)),
                    type(self)(self, indices=list(range(i * block, (i + 1) * block)), filter_name='fold{}[val]'.format(i + 1))
            )

    def __getitem__(self, index):
        if self.indices is None:
            return self.owner_dataset[index]
        return self.owner_dataset[self.indices[index]]

    def __len__(self):
        if self.indices is None:
            return len(self.owner_dataset)
        return len(self.indices)

    def get_metainfo(self, index):
        if self.indices is None:
            return self.owner_dataset.get_metainfo(index)
        return self.owner_dataset.get_metainfo(self.indices[index])

