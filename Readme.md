# Basic U-Net example by MIC@DKFZ
This python package is an example project of how to use a U-Net (Ronneberger et al.) for segmentation on medical images using 
PyTorch (https://www.pytorch.org).
It was developed at the Division of Medical Image Computing at the German Cancer Research Center (DKFZ).
It is also an example on how to use our other python packages batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) and 
Trixi (https://github.com/MIC-DKFZ/trixi) to suit all our deep learning data augmentation needs.
If you encounter a bug, feel free to contact us or open a github issue.

## How to set it up
The example is very easy to use. Just create a new virtual environment in python and install the requirements. 
This example requires python3. We suggest to use virtualenvwrapper (https://virtualenvwrapper.readthedocs.io/en/latest/).
```
mkvirtualenv unet_example
pip3 install -r requirements.txt
```

After setting up the virtual environment you have to start visdom once so it can download some needed files. You only
have to do that once. You can stop the visdom server after a few seconds when it finished downloading the files.
```
python3 -m visdom.server
```

You can edit the paths for data storage and logging in the config file. By default, everything is stored in your working directory.


## How to use it
To start the training simply run 
```
python3 run_train_pipeline.py
```

This will download the Hippocampus dataset from the medical segmentation decathlon (http://medicaldecathlon.com),
extract and preprocess it and then start the training.

If you run the pipeline again, the dataset will not be downloaded, extracted or preprocessed again. To enforce it, just delete the folder.

The training process will automatically be visualized using trixi/visdom. After starting the training you navigate 
in your browser to the port which is printed by the training script. Then you should see your loss curve and so on.

By default, a 2-dimensional U-Net is used. The example also comes with a 3-D version of the network (Özgün Cicek et al.).
To use the 3-D version, simple use
```
python train3D.py
```


## How to use it for your own data
This is work in progress.

### Config

The included `Config_unet.py` is an example config file. You have to adapt this to fit your local environment.

Choose the `#Train parameters` to fit both, your data and your workstation. 
With `fold` you can choose which split from your `splits.pkl` you want to use for the training.

You may also need to adapt the paths (`data_root_dir, data_dir, data_test_dir and split_dir`).

You can change the `Logging parameters`, if you want to. With `append_rnd_string`, you can give each experiment you start a unique name.
If you want to start your visdom server manually, just set `start_visdom=False`.

### Datasets
If you want to use the provided DataLoader, you need to preprocess your data in a appropriate way. An example can be found in the 
"example_dataset" folder. Make sure to load your images and your labels as numpy arrays. The required shape is `(#slices, w,h)`. 
Then save both using:
```
result = np.stack((image, label))

np.save(output_filename, result)
```

The provided DataLoader requires a splits.pkl file, that contains a dictionary of all the files used for training, validation and testing.
It looks like this:
```
[{'train': ['dataset_name_1',...], 'val': ['dataset_name_2', ...], 'test': ['dataset_name_3', ...]}]
```

We use the `MIC/batchgenerators` to perform data augmentation. The example uses cropping, mirroring and some elstic spatial transformation.
You can change the data augmentation by editing the `data_augmentation.py`. Please see the `MIC/batchgenerators` documentation for more details.

To train your network, simply run
```
python train.py
```

You can either edit the config file, or add comandline parameters like this:
```
python train.py --n_epochs 100 [...]
```

## Networks
This example contains a simple implementation of the U-Net (Ronneberger et al.), which can be found in `networks>UNET.py`. 
A little more generic version of the U-Net, as well as the 3D U-Net(Özgün Cicek et al.) can be found in `networks>RecursiveUNet.py` 
respectively `networks>RecursiveUNet3D.py`. This implementation is done in a recursive way.
It is therefor very easy to configure the number of downsamplings. Also the type of normalization can be passed as a parameter (default is
nn.InstanceNorm2d).