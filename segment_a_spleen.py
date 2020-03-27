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
import sys

from medpy.io import save

from configs.Config_unet_spleen import get_config
from datasets.spleen.preprocessing import preprocess_single_file, postprocess_single_image
from experiments.UNetExperiment import UNetExperiment


def save_single_image(image, image_header, filename):
    # medpy.io.save
    save(image, filename, image_header)
    print('> Resulting Image stored as {}'.format(filename))


if __name__ == "__main__":
    c = get_config()

    if len(sys.argv) == 1:
        print("USAGE:\n\npython {} imagefilename [model_checkpoint [shapesize]]\n\n"
              "  imagefilename - a filename that stores a nii.gz formatted file.\n"
              "  model_checkpoint - a checkpoint filename to reload\n"
              "  shapesize - optional value that defines "
              "the size of the shape, default is 64 (not yet used).".format(sys.argv[0]))
        filename = "data/Task09_Spleen/imagesTs/spleen_15.nii.gz"
    else:
        filename = sys.argv[1]

    print("Loading and processing file {}".format(filename))

    if len(sys.argv) > 2:
        c.checkpoint_dir = sys.argv[2]
        c.do_load_checkpoint = True

    print("Loading model from checkpoint {}".format(c.model_dir))
    if len(c.model_dir) == 0 or not os.path.isdir(os.path.split(c.model_dir)[0]):
        print("ERROR /!\\: No checkpoint dir is set, please provide in Config file.")
        exit()

    shapesize = 64
    if len(sys.argv) > 3:
        shapesize = int(sys.argv[3])

    # Get the header in order to preserve voxel dimensions to store the segmented image later on
    print('Preprocessing data.')
    data, header = preprocess_single_file(filename, y_shape=shapesize, z_shape=shapesize)

    print('Setting up model and start segmentation.')
    exp = UNetExperiment(config=c, name=c.name, n_epochs=c.n_epochs,
                         seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals()
                         )

    result = exp.segment_single_image(data)

    print('Postprocessing data.')
    result = postprocess_single_image(result)

    pathname, fname = os.path.split(filename)
    destination_filename = pathname+"/segmented_"+fname
    print('Saving file to disk: {}'.format(destination_filename))
    save_single_image(result, header, destination_filename)
