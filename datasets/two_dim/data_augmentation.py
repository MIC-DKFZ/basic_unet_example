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

from batchgenerators.transforms import Compose, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

import numpy as np


def get_transforms(mode="train", target_size=128):
    tranform_list = []

    if mode == "train":
        tranform_list = [# CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=(target_size,target_size), order=1),
                         MirrorTransform(axes=(1,)),
                         SpatialTransform(patch_size=(target_size, target_size), random_crop=False,
                                          patch_center_dist_from_border=target_size // 2,
                                          do_elastic_deform=True, alpha=(0., 900.), sigma=(20., 30.),
                                          do_rotation=True, p_rot_per_sample=0.8,
                                          angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                          scale=(0.85, 1.25), p_scale_per_sample=0.8,
                                          border_mode_data="nearest", border_mode_seg="nearest"),
                         ]


    elif mode == "val":
        tranform_list = [CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=target_size, order=1),
                         ]

    elif mode == "test":
        tranform_list = [CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=target_size, order=1),
                         ]

    tranform_list.append(NumpyToTensor())

    return Compose(tranform_list)
