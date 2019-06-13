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

import numpy as np
import SimpleITK
import os


def reshape(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0]+x_offset, y_offset:orig_img.shape[1]+y_offset, z_offset:orig_img.shape[2]+z_offset] = orig_img
    # insert temp_img.min() as background value

    return reshaped_image

def save_segmentation(array, reference_dir, output_dir, key):
    image = reference_dir + key.split('/')[-1].replace('npy', 'nii.gz')

    img = SimpleITK.ReadImage(image)

    img_size = img.GetSize()
    # result_reshaped = np.swapaxes(np.asarray(gt_dict[key]).squeeze(), 0, 2)[0:img_size[0], 0:img_size[1], 0:img_size[2]]
    result_reshaped = np.swapaxes(array, 0, 2)[0:img_size[2], 0:img_size[1], 0:img_size[0]]
    image_to_write = SimpleITK.GetImageFromArray(result_reshaped)
    image_to_write.SetSpacing(img.GetSpacing())
    image_to_write.SetOrigin(img.GetOrigin())
    image_to_write.SetDirection(img.GetDirection())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    SimpleITK.WriteImage(SimpleITK.Cast(image_to_write, SimpleITK.sitkUInt8), os.path.join(output_dir,  key.split('/')[-1].replace('npy', 'nii.gz')) + '.nrrd')
