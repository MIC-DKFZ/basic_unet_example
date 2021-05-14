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

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

from networks.UNET import UNet
from loss_functions.dice_loss import SoftDiceLoss


class UNetExperiment(PytorchExperiment):
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
                                              keys=tr_keys)
        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=False)
        self.test_data_loader = NumpyDataSet(self.config.data_test_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                             keys=test_keys, mode="test", do_reshuffle=False)
        self.model = UNet(num_classes=self.config.num_classes, in_channels=self.config.in_channels)

        self.model.to(self.device)

        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax for DICE Loss!
        self.ce_loss = torch.nn.CrossEntropyLoss()  # No softmax for CE Loss -> is implemented in torch!

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model",))

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        data = None
        batch_counter = 0
        for data_batch in self.train_data_loader:

            self.optimizer.zero_grad()

            # Shape of data_batch = [1, b, c, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            data = data_batch['data'][0].float().to(self.device)
            target = data_batch['seg'][0].long().to(self.device)

            pred = self.model(data)
            pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

            loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
            # loss = self.ce_loss(pred, target.squeeze())

            loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: {0} Loss: {1:.4f}'.format(self._epoch_idx, loss))

                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))

                self.clog.show_image_grid(data.float().cpu(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                self.clog.show_image_grid(target.float().cpu(), name="mask", title="Mask", n_iter=epoch)
                self.clog.show_image_grid(pred.cpu()[:, 1:2, ], name="unt", normalize=True, scale_each=True, n_iter=epoch)

            batch_counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'

    def validate(self, epoch):
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch['seg'][0].long().to(self.device)

                pred = self.model(data)
                pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

                loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, float(np.mean(loss_list))))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)

        self.clog.show_image_grid(data.float().cpu(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        self.clog.show_image_grid(target.float().cpu(), name="mask_val", title="Mask", n_iter=epoch)
        self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):
        from evaluation.evaluator import aggregate_scores, Evaluator
        from collections import defaultdict

        self.elog.print('=====TEST=====')
        self.model.eval()

        pred_dict = defaultdict(list)
        gt_dict = defaultdict(list)

        batch_counter = 0
        with torch.no_grad():
            for data_batch in self.test_data_loader:
                print('testing...', batch_counter)
                batch_counter += 1

                # Get data_batches
                mr_data = data_batch['data'][0].float().to(self.device)
                mr_target = data_batch['seg'][0].float().to(self.device)

                pred = self.model(mr_data)
                pred_argmax = torch.argmax(pred.data.cpu(), dim=1, keepdim=True)

                fnames = data_batch['fnames']
                for i, fname in enumerate(fnames):
                    pred_dict[fname[0]].append(pred_argmax[i].detach().cpu().numpy())
                    gt_dict[fname[0]].append(mr_target[i].detach().cpu().numpy())

        test_ref_list = []
        for key in pred_dict.keys():
            test_ref_list.append((np.stack(pred_dict[key]), np.stack(gt_dict[key])))

        scores = aggregate_scores(test_ref_list, evaluator=Evaluator, json_author=self.config.author, json_task=self.config.name, json_name=self.config.name,
                                  json_output_file=self.elog.work_dir + "/{}_".format(self.config.author) + self.config.name + '.json')

        print("Scores:\n", scores)

    def segment_single_image(self, data):
        self.model = UNet(num_classes=self.config.num_classes, in_channels=self.config.in_channels)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # a model must be present and loaded in here
        if self.config.model_dir == '':
            print('model_dir is empty, please provide directory to load checkpoint.')
        else:
            self.load_checkpoint(name=self.config.model_dir, save_types=("model",))

        self.elog.print("=====SEGMENT_SINGLE_IMAGE=====")
        self.model.eval()
        self.model.to(self.device)

        # Desired shape = [b, c, w, h]
        # split into even chunks (lets use size)
        with torch.no_grad():

            ######
            # When working entirely on CPU and in memory, the following lines replace the split/concat method
            # mr_data = data.float().to(self.device)
            # pred = self.model(mr_data)
            # pred_argmax = torch.argmax(pred.data.cpu(), dim=1, keepdim=True)
            ######

            ######
            # for CUDA (also works on CPU) split into batches
            blocksize = self.config.batch_size

            # number_of_elements = round(data.shape[0]/blocksize+0.5)     # make blocks large enough to not lose any slices
            chunks = [data[i:i+blocksize, ::, ::, ::] for i in range(0, data.shape[0], blocksize)]
            pred_list = []
            for data_batch in chunks:
                mr_data = data_batch.float().to(self.device)
                pred_dict = self.model(mr_data)
                pred_list.append(pred_dict.cpu())

            pred = torch.Tensor(np.concatenate(pred_list))
            pred_argmax = torch.argmax(pred, dim=1, keepdim=True)

        # detach result and put it back to cpu so that we can work with, create a numpy array
        result = pred_argmax.short().detach().cpu().numpy()

        return result
