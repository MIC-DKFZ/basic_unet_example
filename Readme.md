# Basic_UNet example by MIC@DKFZ
This python package is an example project of how to use a UNet (Ronneberger et al.) for segmentation on medical images.
It was developed at the Division of Medical Image Computing at the German Cancer
Research Center (DKFZ).
It is also an example on how to use our other python packages batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) and Trixi (https://trixi.readthedocs.io) to suit all our deep learning data augmentation needs.
If you encounter bug, feel free to contact us or open a github issue.


## How to set it up
The example is very easy to use. Just create a new virtual environment in python and install the requirements. We suggest to use virtualenvwrapper (https://virtualenvwrapper.readthedocs.io/en/latest/).
```
mkvirtualenv unet_example
pip install -r requirements.txt
```

Edit the configs/Config_unet.py:
```
# Adapt these paths for your environment!
c.base_dir = './output/unet_example/'  # Where to log the output of the experiment.

c.data_root_dir = './example_unet_dataset/'  # The path where the downloaded dataset is stored.
```

## How to use it
To start the training simply run 
```
python run_train_pipeline.py
```

This will download the Hippocampus dataset from the medical segmentation decathlon (http://medicaldecathlon.com),
extract and preprocess it and then start the training.

## Detailed description
Todo