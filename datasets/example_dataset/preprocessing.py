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

from collections import defaultdict

from batchgenerators.augmentations.utils import pad_nd_image
from medpy.io import load
import os
import numpy as np
import torch

from utilities.file_and_folder_operations import subfiles


def preprocess_data(root_dir, y_shape=64, z_shape=64):
    image_dir = os.path.join(root_dir, 'imagesTr')
    label_dir = os.path.join(root_dir, 'labelsTr')
    output_dir = os.path.join(root_dir, 'preprocessed')
    classes = 3

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    class_stats = defaultdict(int)
    total = 0

    nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

    for i in range(0, len(nii_files)):
        if nii_files[i].startswith("._"):
            nii_files[i] = nii_files[i][2:]

    for f in nii_files:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f.replace('_0000', '')))

        print(f)

        for i in range(classes):
            class_stats[i] += np.sum(label == i)
            total += np.sum(label == i)

        # normalize images
        image = (image - image.min())/(image.max()-image.min())

        image = pad_nd_image(image, (image.shape[0], y_shape, z_shape), "constant", kwargs={'constant_values': image.min()})
        label = pad_nd_image(label, (image.shape[0], y_shape, z_shape), "constant", kwargs={'constant_values': label.min()})

        result = np.stack((image, label))

        np.save(os.path.join(output_dir, f.split('.')[0]+'.npy'), result)
        print(f)

    print(total)
    for i in range(classes):
        print(class_stats[i], class_stats[i]/total)


def preprocess_single_file(image_file):
    image, image_header = load(image_file)
    image = (image - image.min()) / (image.max() - image.min())

    data = np.expand_dims(image, 1)

    return torch.from_numpy(data), image_header


def postprocess_single_image(image):
    # desired shape is [b w h]
    result_converted = image[::, 0, ::, ::]
    result_mapped = [i * 255 for i in result_converted]

    return result_mapped
