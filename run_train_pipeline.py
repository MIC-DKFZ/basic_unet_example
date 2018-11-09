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

from configs.Config_unet import get_config
from datasets.example_dataset.create_splits import create_splits
from datasets.example_dataset.download_dataset import download_dataset
from datasets.example_dataset.preprocessing import preprocess_data
from experiments.UNetExperiment import UNetExperiment

c = get_config()

dataset_name = 'Task04_Hippocampus'
download_dataset(dest_path=c.data_root_dir, dataset=dataset_name)

preprocess_data(root_dir=os.path.join(c.data_root_dir, dataset_name))
create_splits(output_dir=c.split_dir, image_dir=c.data_dir)

exp = UNetExperiment(config=c, name='unet_experiment', n_epochs=c.n_epochs,
                     seed=42, append_rnd_to_name=c.append_rnd_string, visdomlogger_kwargs={"auto_start": c.start_visdom})

exp.run()
# exp.run_test(setup=False)