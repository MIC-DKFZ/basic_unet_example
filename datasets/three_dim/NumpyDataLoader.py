#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fnmatch
import random

import numpy as np

from batchgenerators.dataloading import SlimDataLoaderBase
from datasets.data_loader import MultiThreadedDataLoader
from .data_augmentation import get_transforms


def load_dataset(base_dir, pattern='*.npy', keys=None):
    fls = []
    files_len = []
    dataset = []

    for root, dirs, files in os.walk(base_dir):
        i = 0
        for filename in sorted(fnmatch.filter(files, pattern)):

            if keys is not None and filename[:-4] in keys:
                npy_file = os.path.join(root, filename)
                numpy_array = np.load(npy_file, mmap_mode="r")

                fls.append(npy_file)
                files_len.append(numpy_array.shape[1])

                dataset.extend([i])

                i += 1

    return fls, files_len, dataset


class NumpyDataSet(object):
    """
    TODO
    """
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000, seed=None, num_processes=8, num_cached_per_queue=8 * 4, target_size=128,
                 file_pattern='*.npy', label=1, input=(0,), do_reshuffle=True, keys=None):

        data_loader = NumpyDataLoader(base_dir=base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches, seed=seed, file_pattern=file_pattern,
                                      input=input, label=label, keys=keys)

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms, num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue, seeds=seed,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)


class NumpyDataLoader(SlimDataLoaderBase):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 seed=None, file_pattern='*.npy', label=1, input=(0,), keys=None):

        self.files, self.file_len, self.dataset = load_dataset(base_dir=base_dir, pattern=file_pattern, keys=keys, )
        super(NumpyDataLoader, self).__init__(self.dataset, batch_size, num_batches)

        self.batch_size = batch_size

        self.use_next = False
        if mode == "train":
            self.use_next = False

        self.idxs = list(range(0, len(self.dataset)))

        self.data_len = len(self.dataset)

        self.num_batches = min((self.data_len // self.batch_size)+10, num_batches)

        if isinstance(label, int):
            label = (label,)
        self.input = input
        self.label = label

        self.np_data = np.asarray(self.dataset)

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.idxs)
        print("Initializing... this might take a while...")

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        return self.get_data_from_array(open_arr)

    def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items

    def __getitem__(self, item):
        idxs = self.idxs
        data_len = len(self.dataset)
        np_data = self.np_data

        if item > len(self):
            raise StopIteration()
        if (item * self.batch_size) == data_len:
            raise StopIteration()

        start_idx = (item * self.batch_size) % data_len
        stop_idx = ((item + 1) * self.batch_size) % data_len

        if ((item + 1) * self.batch_size) == data_len:
            stop_idx = data_len

        if stop_idx > start_idx:
            idxs = idxs[start_idx:stop_idx]
        else:
            raise StopIteration()

        open_arr = np_data[idxs]

        return self.get_data_from_array(open_arr)

    def get_data_from_array(self, open_array):
        data = []
        fnames = []
        idxs = []
        labels = []

        for idx in open_array:
            fn_name = self.files[idx]

            numpy_array = np.load(fn_name, mmap_mode="r")

            data.append(numpy_array[list(self.input)])   # 'None' keeps the dimension

            if self.label is not None:
                labels.append(numpy_array[list(self.input)])   # 'None' keeps the dimension

            fnames.append(self.files[idx])
            idxs.append(idx)

        ret_dict = {'data': data, 'fnames': fnames, 'idxs': idxs}
        if self.label is not None:
            ret_dict['seg'] = labels

        return ret_dict
