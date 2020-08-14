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
    """File to split the dataset into multiple folds and the train, validation and test set.

    :param output_dir: Directory to write the splits file to
    :param image_dir: Directory where the images lie in.
    """
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    sample_size = len(npy_files)

    testset_size = int(sample_size * 0.25)
    valset_size = int(sample_size * 0.25)
    trainset_size = sample_size - valset_size - testset_size  # Assure all samples are used.

    if sample_size < (trainset_size + valset_size + testset_size):
        raise ValueError("Assure more total samples exist than train test and val samples combined!")

    splits = []
    sample_set = {sample[:-4] for sample in npy_files.copy()}  # Remove the file extension
    test_samples = random.sample(sample_set, testset_size)  # IMO the Testset should be static for all splits

    for split in range(0, 5):
        train_samples = random.sample(sample_set - set(test_samples), trainset_size)
        val_samples = list(sample_set - set(train_samples) - set(test_samples))

        train_samples.sort()
        val_samples.sort()

        split_dict = dict()
        split_dict['train'] = train_samples
        split_dict['val'] = val_samples
        split_dict['test'] = test_samples

        splits.append(split_dict)

    # Todo: IMO it is better to write that dict as JSON. This (unlike pickle) allows the user to inspect the file with an editor
    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)

    splits_sanity_check(output_dir)


# ToDo: The naming "splits.pkl should not be distributed over multiple files. This makes changing of it less clear.
#   Instead move saving and loading to one file. (Here would be a good place)
#   Other usages are: spleen/create_splits.py:57 (Which is redundand anyways?);
#   UNetExperiment3D.py:55  and UNetExperiment.py:55
def splits_sanity_check(path):
    """ Takes path to a splits file and verifies that no samples from the test dataset leaked into train or validation.
    :param path
    """
    with open(os.path.join(path, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
        for i in range(len(splits)):
            samples = splits[i]
            tr_samples = set(samples["train"])
            vl_samples = set(samples["val"])
            ts_samples = set(samples["test"])

            assert len(tr_samples.intersection(vl_samples)) == 0, "Train and validation samples overlap!"
            assert len(vl_samples.intersection(ts_samples)) == 0, "Validation and Test samples overlap!"
            assert len(tr_samples.intersection(ts_samples)) == 0, "Train and Test samples overlap!"
    return
