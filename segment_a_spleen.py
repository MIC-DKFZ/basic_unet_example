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
from os.path import exists
import sys

from medpy.io import load
from medpy.io import save
import numpy as np
from datasets.utils import reshape

from configs.Config_unet_spleen_exec import get_config
from experiments.UNetExperiment import UNetExperiment
import torch


def preprocess_single_file(image_file, y_shape=64, z_shape=64):
    image, image_header = load(image_file)
    image = (image - image.min()) / (image.max() - image.min())

    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)

    # TODO check original shape and reshape data if necessary
    # image = reshape(image, append_value=0, new_shape=(image.shape[0], y_shape, z_shape))
    # numpy_array = np.array(image)

    # Image shape is [b, w, h] and has one channel only
    # Desired shape = [b, c, w, h]
    # --> expand to have only one channel c=1 - data is in desired shape
    data = np.expand_dims(image, 1)

    return torch.from_numpy(data), image_header


def postprocess_single_image(image):
    #desired shape is [b w h]
    result_converted = image[::, 0, ::, ::]
    result_mapped = [i * 255 for i in result_converted]

    # swap axes back, like we were supposed to do so
    result_mapped = np.swapaxes(result_mapped, 2, 1)
    result_mapped = np.swapaxes(result_mapped, 2, 0)

    return result_mapped


def save_single_image(image, image_header, filename):
    # medpy.io.save
    save(image, filename, image_header)
    print('Result Image stored as {}'.format(filename))


if __name__ == "__main__":
    c = get_config()

    if len(sys.argv) == 1:
        print("USAGE:\n\npython {} imagefilename [model_checkpoint [shapesize]]\n\n"
              "  imagefilename - a filename that stores a nii.gz formatted file.\n"
              "  model_checkpoint - a checkpoint filename to reload\n"
              "  shapesize - optional value that defines "
              "the size of the shape, default is 64 (not yet used).".format(sys.argv[0]))
        filename = "../spleen_8.nii.gz"
    else:
        filename = sys.argv[1]

    print("Loading and processing file {}".format(filename))

    if len(sys.argv) > 2:
        c.checkpoint_dir = sys.argv[2]
        c.do_load_checkpoint = True

    if c.do_load_checkpoint:
        print("Loading checkpoint from file {}".format(c.checkpoint_dir))
        if len(c.checkpoint_dir) == 0 or not os.path.isdir(os.path.split(c.checkpoint_dir)[0]):
            print("WARNING /!\\: No checkpoint dir is set, please provide in Config file.")

    shapesize = 64
    if len(sys.argv) > 3:
        shapesize = int(sys.argv[3])

    # Get the header in order to preserve voxel dimensions to store the segmented image later on
    print('Preprocessing data. [STARTED]')
    data, header = preprocess_single_file(filename, y_shape=shapesize, z_shape=shapesize)
    print('Preprocessing data. [DONE]')

    exp = UNetExperiment(config=c, name=c.name, n_epochs=c.n_epochs,
                         seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals(),
                         # visdomlogger_kwargs={"auto_start": c.start_visdom},
                         #loggers={
                         #    "visdom": ("visdom", {"auto_start": c.start_visdom})
                         #}
                         )

    result = exp.segment_single_image(data)

    print('Postprocessing data. [STARTED]')
    result = postprocess_single_image(result)
    print('Postprocessing data. [ENDED]')

    pathname, fname = os.path.split(filename)
    save_single_image(result, header, pathname+"/segmented_"+fname)
