import os
import pickle

from utilities.file_and_folder_operations import subfiles

root_dir = '/workplace/data/3D/20200313_optical_prop_rough'
output_dir = '/workplace/data/3D'

files = subfiles(root_dir, join=False)

train_files = files[0:800]
val_files = files[800:900]
test_files = files[900:1000]

# train_files = files[0:800]
# val_files = files[801:900]
# test_files = files[901:1000]

split_dict = {}
split_dict['train'] = train_files
split_dict['val'] = val_files
split_dict['test'] = test_files

splits = []
splits.append(split_dict)

with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
    pickle.dump(splits, f)
print('bla')
