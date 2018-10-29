from collections import defaultdict

from medpy.io import load
import os
import numpy as np

from datasets.utils import reshape
from utilities.file_and_folder_operations import subfiles

root_dir = "/media/kleina/Data/Data/Hippocampus"
image_dir = os.path.join(root_dir, 'imagesTr')
label_dir = os.path.join(root_dir, 'labelsTr')
output_dir = "/media/kleina/Data/Data/Hippocampus/preprocessed"

classes = 3

class_stats = defaultdict(int)
total = 0

nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

for f in nii_files:
    image, _ = load(os.path.join(image_dir, f))
    label, _ = load(os.path.join(label_dir, f.replace('_0000', '')))

    print(f)
    print(image.shape)

    for i in range(classes):
        class_stats[i] += np.sum(label == i)
        total += np.sum(label == i)

    image = (image - image.min())/(image.max()-image.min())

    reshaped_image = reshape(image, (1, 1, 1), append_value=0, new_shape=(64, 64, 64))
    reshaped_label = reshape(label, (1, 1, 1), append_value=0, new_shape=(64, 64, 64))
    result = np.stack((reshaped_image, reshaped_label))

    np.save(os.path.join(output_dir, f.split('.')[0]+'.npy'), result)
    print(f)

print(total)
for i in range(classes):
    print(class_stats[i], class_stats[i]/total)
