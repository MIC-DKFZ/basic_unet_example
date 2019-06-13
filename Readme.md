# Basic U-Net example by MIC@DKFZ
This python package is an example project of how to use a U-Net [1] for segmentation on medical images using 
PyTorch (https://www.pytorch.org).
It was developed at the Division of Medical Image Computing at the German Cancer Research Center (DKFZ).
It is also an example on how to use our other python packages batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) and 
Trixi (https://github.com/MIC-DKFZ/trixi) [2] to suit all our deep learning data augmentation needs.

If you have any questions or issues or you encounter a bug, feel free to contact us, open a github issue or ask the community on Gitter:
[![Gitter](https://badges.gitter.im/basic-Unet/community.svg)](https://gitter.im/basic-Unet/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


## How to set it up
The example is very easy to use. Just create a new virtual environment in python and install the requirements. 
This example requires python3. It was implemented with python 3.5. We suggest to use virtualenvwrapper (https://virtualenvwrapper.readthedocs.io/en/latest/).
```
mkvirtualenv unet_example
pip3 install -r requirements.txt
```

In this example code we show how to use visdom for live visualization. See the Trixi documentation for more details or information about other tools like tensorboard.
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
extract and preprocess it and then start the training. The preprocessing loads the images (imagesTr) and the corresponding labels (labelsTr), performes some 
normalization and padding operations and saves the data as NPY files. The available images are then split into `train`, `validation` and `test` sets.
The splits are saved to a `splits.pkl` file. The images in `imagesTs` are  not used in the example, because they are the test set for the 
medical segmentation decathlon and therefor no ground truth is provided.

If you run the pipeline again, the dataset will not be downloaded, extracted or preprocessed again. To enforce it, just delete the folder.

The training process will automatically be visualized using trixi/visdom. After starting the training you navigate 
in your browser to the port which is printed by the training script. Then you should see your loss curve and so on.

By default, a 2-dimensional U-Net is used. The example also comes with a 3-D version of the network (Özgün Cicek et al.).
To use the 3-D version, simple use
```
python train3D.py
```

<aside class="warning">
The 3-D version is not yet tested thoroughly. Use it with caution!
</aside>

## How to use it for your own data
This description is work in progress. If you use this repo for your own data please share your experience, so we can update this part.

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

We use the `MIC/batchgenerators` to perform data augmentation. The example uses cropping, mirroring and some elastic spatial transformation.
You can change the data augmentation by editing the `data_augmentation.py`. Please see the `MIC/batchgenerators` documentation for more details.

To train your network, simply run
```
python train.py
```

You can either edit the config file, or add commandline parameters like this:
```
python train.py --n_epochs 100 [...]
```

## Networks
This example contains a simple implementation of the U-Net [1], which can be found in `networks>UNET.py`. 
A little more generic version of the U-Net, as well as the 3D U-Net [3] can be found in `networks>RecursiveUNet.py` 
respectively `networks>RecursiveUNet3D.py`. This implementation is done in a recursive way.
It is therefor very easy to configure the number of downsamplings. Also the type of normalization can be passed as a parameter (default is
nn.InstanceNorm2d).

## References
[1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." 
International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
[2] David Zimmerer, Jens Petersen, GregorKoehler, Jakob Wasserthal, dzimmm, Tim, … André Pequeño. (2018, November 23). MIC-DKFZ/trixi: Alpha (Version v0.1.1). 
Zenodo. http://doi.org/10.5281/zenodo.1495180
[3] Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation." 
International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.

