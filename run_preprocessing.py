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
from datasets.utils import download_dataset
from datasets.example_dataset.preprocessing import preprocess_data

if __name__ == "__main__":
    c = get_config()

    download_dataset(dest_path=c.data_root_dir, dataset=c.dataset_name, id=c.google_drive_id)

    print('Preprocessing data. [STARTED]')
    preprocess_data(root_dir=os.path.join(c.data_root_dir, c.dataset_name))
    create_splits(output_dir=c.split_dir, image_dir=c.data_dir)
    print('Preprocessing data. [DONE]')
