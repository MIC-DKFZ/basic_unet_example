import os

import torch
from trixi.util import Config


def get_config():
    c = Config()

    if torch.cuda.is_available():
        c.use_cuda = True
    else:
        c.use_cuda = False

    c.batch_size = 6
    c.patch_size = 64
    c.n_epochs = 20
    c.learning_rate = 0.0002
    c.fold = 1

    c.do_batchnorm = True  # Defines whether or not the UNet does a batchnorm in the contracting path
    c.do_load_checkpoint = False
    c.checkpoint_dir = ''

    c.unpack_data = True

    # Adapt these paths for your environment!
    c.base_dir = '/media/kleina/Data/output/unet_example/'  # Where to log the output of the experiment.

    c.data_dir = '/media/kleina/Data/Data/Hippocampus/preprocessed'  # This is where your training and validation data is stored
    c.data_test_dir = '/media/kleina/Data/Data/Hippocampus/preprocessed'  # This is where your test data is stored

    c.split_dir = '/media/kleina/Data/Data/Hippocampus/' # This is where the 'splits.pkl' file is located, that holds your splits.
    c.additional_slices = 0
    c.name = 'Basic_seg'

    c.plot_freq = 10  # How often should stuff be shown in visdom
    c.append_rnd_string = False
    c.start_visdom = False

    print(c)
    return c
