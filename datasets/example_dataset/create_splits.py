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

import pickle
from utilities.file_and_folder_operations import subfiles

import os
import random


def create_splits(output_dir, image_dir):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    sample_size = len(npy_files)

    trainset_size = sample_size*50//100
    valset_size = sample_size*25//100
    testset_size = sample_size*25//100

    if sample_size < (trainset_size + valset_size + testset_size):
        raise ValueError("Assure more total samples exist than train test and val samples combined!")

    splits = []
    for split in range(0, 5):
        image_list = npy_files.copy()
        sample_set = {sample[:-4] for sample in image_list}  # Remove the file extension

        train_set = set(random.sample(sample_set, trainset_size))
        val_set = set(random.sample(sample_set - train_set, valset_size))
        test_set = set(random.sample(sample_set - train_set - val_set, testset_size))

        split_dict = dict()
        split_dict['train'] = list(train_set)
        split_dict['val'] = list(val_set)
        split_dict['test'] = list(test_set)

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
