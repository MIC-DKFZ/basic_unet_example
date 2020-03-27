#config for simulated data for 2D

import os
import numpy as np
import fnmatch
from shutil import copyfile

base_dir = '/home/melanie/networkdrives/E130-Projekte/Photoacoustics/RawData/20200306_optical_prop_rough'
output_dir = '/workplace/data/2D/20200306_optical_prop_rough'
pattern= '*.npz'
keys_data='properties_700nm'

detailed_base_dir_data=[]
for root, dirs, files in os.walk(base_dir):
    i = 0
    # print(files)
    for filename in sorted(fnmatch.filter(files, pattern)):
        # print(filename)
        if keys_data is not None and filename[:-4] in keys_data:  # - because we want to learn 1 wavelength at the moment eg optical_forward_model_output_660
            detailed_base_dir_data.append(os.path.join(root, filename))
detailed_base_dir_data = sorted(detailed_base_dir_data)



keys_target='optical_forward_model_output_700'

detailed_base_dir_target=[]
for root, dirs, files in os.walk(base_dir):
    i = 0
    # print(files)
    for filename in sorted(fnmatch.filter(files, pattern)):
        # print(filename)
        if keys_target is not None and filename[:-4] in keys_target:  # - because we want to learn 1 wavelength at the moment eg optical_forward_model_output_660
            detailed_base_dir_target.append(os.path.join(root, filename))
detailed_base_dir_target = sorted(detailed_base_dir_target)

fileparams_data = ('mua', 'mus', 'g')
fileparams_target = ('initial_pressure',)

dataset = []
for i in range(0, len(detailed_base_dir_data)):
    print(i)
    numpy_array_dict = np.load(detailed_base_dir_data[i])
    channels = len(fileparams_data)
    array_with_channels = np.empty([4,34,34])
#        [4, numpy_array_dict.__getitem__(fileparams_data[0]).shape[0], numpy_array_dict.__getitem__(fileparams_data[0]).shape[1]])

    for idc in range(0, len(fileparams_data)):
        array_to_save = numpy_array_dict.__getitem__(fileparams_data[idc])
        array_with_channels[idc, :, :] = array_to_save[:, 10, 1:]

    numpy_array_dict = np.load(detailed_base_dir_target[i])
    if len(fileparams_target) == 1:
        array_to_save_tar = numpy_array_dict.__getitem__(fileparams_target[0])

        #array_to_save_tar_new=array_to_save_tar/np.max(array_to_save_tar)*500
        #x=np.arange(0,500)
        #inds=np.digitize(array_to_save_tar_new, x)
        array_with_channels[3, :, :] = array_to_save_tar[:, 10, 1:]
        #array_with_channels[3, :, :] = array_to_save_tar
    # fls.append(array_with_channels)

    #array_with_channels = np.transpose(array_with_channels, (0, 3, 1, 2))
    array_with_channels = np.expand_dims(array_with_channels, 1) #dimensions c x 1 x dim1 x dim2

    #Be careful here
    dst = os.path.join(output_dir, detailed_base_dir_data[i].split('/')[-2] + '.npy')


    np.save(dst, array_with_channels)

