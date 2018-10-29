import os

import torch
from trixi.util import Config


def get_config():
    c = Config()

    if torch.cuda.is_available():
        c.use_cuda = True
    else:
        c.use_cuda = False

    # Train parameters
    c.batch_size = 8
    c.patch_size = 64
    c.n_epochs = 100
    c.learning_rate = 0.0002
    c.fold = 0  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

    # Logging parameters
    c.name = 'Basic_seg'
    c.plot_freq = 10  # How often should stuff be shown in visdom
    c.append_rnd_string = False
    c.start_visdom = True

    c.do_batchnorm = True  # Defines whether or not the UNet does a batchnorm in the contracting path
    c.do_load_checkpoint = False
    c.checkpoint_dir = ''

    # Adapt these paths for your environment!
    c.base_dir = '/media/kleina/Data/output/unet_example/'  # Where to log the output of the experiment.

    c.data_root_dir = '/media/kleina/Data/Data/example_unet_dataset/'
    c.data_dir = os.path.join(c.data_root_dir, 'Task04_Hippocampus/preprocessed')  # This is where your training and validation data is stored
    c.data_test_dir = os.path.join(c.data_root_dir, 'Task04_Hippocampus/preprocessed')  # This is where your test data is stored

    c.split_dir = os.path.join(c.data_root_dir, 'Task04_Hippocampus')   # This is where the 'splits.pkl' file is located, that holds your splits.

    print(c)
    return c
