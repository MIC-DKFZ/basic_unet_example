import pickle
from collections import defaultdict

import os
import numpy as np

root_dir = "/media/kleina/Data/Data/Hippocampus"
image_dir = os.path.join(root_dir, 'preprocessed')

output_dir = root_dir

from utilities.file_and_folder_operations import subfiles
npy_files = subfiles(image_dir, suffix=".npy", join=False)

trainset_size = len(npy_files)*50//100
valset_size = len(npy_files)*25//100
testset_size = len(npy_files)*25//100

splits=[]
for split in range(0,5):
    image_list = npy_files.copy()
    trainset = []
    valset = []
    testset = []
    for i in range(0, trainset_size):
        patient = np.random.choice(image_list)
        image_list.remove(patient)
        trainset.append(patient[:-4])
    for i in range(0, valset_size):
        patient = np.random.choice(image_list)
        image_list.remove(patient)
        valset.append(patient[:-4])
    for i in range(0, testset_size):
        patient = np.random.choice(image_list)
        image_list.remove(patient)
        testset.append(patient[:-4])
    split_dict = dict()
    split_dict['train'] = trainset
    split_dict['val'] = valset
    split_dict['test'] = testset

    splits.append(split_dict)

with open(os.path.join(root_dir, 'splits.pkl'), 'wb') as f:
    pickle.dump(splits, f)