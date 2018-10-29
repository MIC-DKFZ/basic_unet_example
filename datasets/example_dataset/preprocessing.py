from collections import defaultdict

from medpy.io import load
import os
import numpy as np

from datasets.utils import reshape
from utilities.file_and_folder_operations import subfiles


def preprocess_data(root_dir):
    image_dir = os.path.join(root_dir, 'imagesTr')
    label_dir = os.path.join(root_dir, 'labelsTr')
    output_dir = os.path.join(root_dir, 'preprocessed')
    classes = 3

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    class_stats = defaultdict(int)
    total = 0

    nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

    for i in range(0, len(nii_files)):
        if nii_files[i].startswith("._"):
            nii_files[i] = nii_files[i][2:]

    for f in nii_files:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f.replace('_0000', '')))

        print(f)
        print(image.shape)

        for i in range(classes):
            class_stats[i] += np.sum(label == i)
            total += np.sum(label == i)

        image = (image - image.min())/(image.max()-image.min())

        image = reshape(image, (1, 1, 1), append_value=0, new_shape=(64, 64, 64))
        label = reshape(label, (1, 1, 1), append_value=0, new_shape=(64, 64, 64))

        result = np.stack((image, label))

        np.save(os.path.join(output_dir, f.split('.')[0]+'.npy'), result)
        print(f)

    print(total)
    for i in range(classes):
        print(class_stats[i], class_stats[i]/total)