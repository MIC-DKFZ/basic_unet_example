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
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.three_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

from networks.RecursiveUNet3D import UNet3D
from loss_functions.dice_loss import SoftDiceLoss, DC_and_CE_loss, MultipleOutputLoss
from loss_functions import MSE_L1_Loss
import apex.amp as amp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt





class UNetExperiment3D(PytorchExperiment):
    """
    The UnetExperiment is inherited from the PytorchExperiment. It implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    It is optimized to work with the provided NumpyDataLoader.

    The basic life cycle of a UnetExperiment is the same s PytorchExperiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()
    """

    def setup(self):
        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        tr_keys = splits[self.config.fold]['train']
        val_keys = splits[self.config.fold]['val']
        test_keys = splits[self.config.fold]['test']

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                              keys=tr_keys, mode="train", do_reshuffle=False, label=None, input=(0, 1, 2, 3,))
        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=False,  label=None, input=(0, 1, 2, 3,))
        self.test_data_loader = NumpyDataSet(self.config.data_test_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                             keys=test_keys, mode="test", do_reshuffle=False,label=None, input=(0, 1, 2, 3,))
        self.model = UNet3D(num_classes=1, in_channels=self.config.in_channels,  num_downs=4)

        self.model.to(self.device)




        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.loss = MSE_L1_loss(aggregate="sum")
        #self.loss = nn.L1Loss()

        #DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
         #                          'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, OrderedDict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model"))

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        batch_counter = 0
        for data_batch in self.train_data_loader:
            self.optimizer.zero_grad()

            # Shape of data_batch = [1, b, c, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            data = data_batch['data'][0, :, 0:3, :, :, :].float().to(self.device)
            target = data_batch['data'][0, :, 3:4, :, :, :].float().to(self.device)

            pred = self.model(data)

            loss = self.loss(pred, target)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            #loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, loss))

                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))

                self.clog.show_image_grid(data[:, 0:1, :, 15, :].float(), name="mua", image_args=dict(normalize=True, scale_each=True, padding=0))
                self.clog.show_image_grid(data[:, 1:2, :, 15, :].float(), name="mus", image_args=dict(normalize=True, scale_each=True, padding=0))
                self.clog.show_image_grid(data[:, 2:3, :, 15, :].float(), name="g", image_args=dict(normalize=True, scale_each=True, padding=0))
                self.clog.show_image_grid(target[:, :, :, 15, :].float(), name="p0", image_args=dict(normalize=True, scale_each=True, padding=0))
                self.clog.show_image_grid(pred[:, :, :, 15, :].float().cpu(), name="pred_p0", image_args=dict(normalize=True, scale_each=True, padding=0))


            batch_counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'

    def validate(self, epoch):
        #if epoch % 5 != 0:
         #   return
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                    data = data_batch['data'][0, :, 0:3, :, :, :].float().to(self.device)
                    target = data_batch['data'][0, :, 3:4, :, :, :].float().to(self.device)

                    pred = self.model(data)

                    loss = self.loss(pred, target)
                    loss_list.append(loss.item())
        
        #find_bad_data 
        #change_pickle_file
        #update_train_data


        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, np.mean(loss_list)))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)

        self.clog.show_image_grid(data[:, 0:1, :, 15, :].float(), name="mua_val", image_args=dict(normalize=True, scale_each=True, padding=0))
        self.clog.show_image_grid(data[:, 1:2, :, 15, :].float(), name="mus_val", image_args=dict(normalize=True, scale_each=True, padding=0))
        self.clog.show_image_grid(data[:, 2:3, :, 15, :].float(), name="g_val", image_args=dict(normalize=True, scale_each=True, padding=0))
        self.clog.show_image_grid(target[:, :, :, 15, :].float(), name="p0_val", image_args=dict(normalize=True, scale_each=True, padding=0))
        self.clog.show_image_grid(pred[:, :, :, 15, :].float().cpu(), name="pred_p0_val", image_args=dict(normalize=True, scale_each=True, padding=0))

    def test(self):
        # TODO
        print('TODO: Implement your test() method here')
