import os

from trixi.util import Config


def get_config():
    # TODO
    data_root_dir = os.path.abspath('data')  # The path where the downloaded dataset is stored.

    c = Config(
        update_from_argv=True,

        # Train parameters
        batch_size=32,
        patch_size=64,
        n_epochs=10,
        learning_rate=0.0002,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # TODO

        # Logging parameters
        name='Basic_Unet',
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        start_visdom=True,

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=False,
        checkpoint_dir='',

        # Adapt this paths for your environment!
        base_dir=os.path.abspath('output_experiment'),  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, 'Task04_Hippocampus/preprocessed'),  # This is where your training and validation data is stored
        data_test_dir=os.path.join(data_root_dir, 'Task04_Hippocampus/preprocessed'),  # This is where your test data is stored

        split_dir=os.path.join(data_root_dir, 'Task04_Hippocampus')  # This is where the 'splits.pkl' file is located, that holds your splits.
    )

    print(c)
    return c
