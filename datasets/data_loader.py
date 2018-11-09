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

from torch.utils.data import DataLoader, Dataset
from trixi.util.pytorchutils import set_seed


class WrappedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.transform = transform
        self.dataset = dataset

        self.is_indexable = False
        if hasattr(self.dataset, "__getitem__") and not (hasattr(self.dataset, "use_next") and self.dataset.use_next is True):
            self.is_indexable = True

    def __getitem__(self, index):

        if not self.is_indexable:
            item = next(self.dataset)
        else:
            item = self.dataset[index]
        item = self.transform(**item)
        return item

    def __len__(self):
        return int(self.dataset.num_batches)


class MultiThreadedDataLoader(object):
    def __init__(self, data_loader, transform, num_processes, **kwargs):

        self.cntr = 1
        self.ds_wrapper = WrappedDataset(data_loader, transform)

        self.generator = DataLoader(self.ds_wrapper, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                                    num_workers=num_processes, pin_memory=True, drop_last=False,
                                    worker_init_fn=self.get_worker_init_fn())

        self.num_processes = num_processes
        self.iter = None

    def get_worker_init_fn(self):
        def init_fn(worker_id):
            set_seed(worker_id + self.cntr)

        return init_fn

    def __iter__(self):
        self.kill_iterator()
        self.iter = iter(self.generator)
        return self.iter

    def __next__(self):
        if self.iter is None:
            self.iter = iter(self.generator)
        return next(self.iter)

    def renew(self):
        self.cntr += 1
        self.kill_iterator()
        self.generator.worker_init_fn = self.get_worker_init_fn()
        self.iter = iter(self.generator)

    def restart(self):
        pass
        # self.iter = iter(self.generator)

    def kill_iterator(self):
        try:
            if self.iter is not None:
                self.iter._shutdown_workers()
                for p in self.iter.workers:
                    p.terminate()
        except:
            print("Could not kill Dataloader Iterator")
