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

from configs.Config_unet import get_config
from experiments.UNetExperiment3D import UNetExperiment3D

if __name__ == "__main__":
    c = get_config()

    exp = UNetExperiment3D(config=c, name=c.name, n_epochs=c.n_epochs,
                         seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals(),
                         # visdomlogger_kwargs={"auto_start": c.start_visdom},
                         loggers={
                             "visdom": ("visdom", {"auto_start": c.start_visdom}),
                             # "tb": ("tensorboard"),
                             # "slack": ("slack", {"token": "XXXXXXXX",
                             #                     "user_email": "x"})
                         }
                         )
    exp.run()
    exp.run_test(setup=False)
