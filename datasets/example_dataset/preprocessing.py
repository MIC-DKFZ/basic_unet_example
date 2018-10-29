from collections import defaultdict

from medpy.io import load
import os
import numpy as np

from datasets.utils import reshape

min_HU = -160
max_HU = 240

root_dir= "/media/kleina/Data/Data/Hippocampus"
image_dir= os.path.join(root_dir, 'imagesTr')
label_dir= os.path.join(root_dir, 'labelsTr')
output_dir = "/media/kleina/Data/Data/Hippocampus/preprocessed"

classes = 3

class_stats = defaultdict(int)
total = 0

from utilities.file_and_folder_operations import subfiles
nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

for f in nii_files:
    image, _ = load(os.path.join(image_dir, f))
    label, _ = load(os.path.join(label_dir, f.replace('_0000', '')))

    # image = np.transpose(image, (2, 0, 1))
    # label = np.transpose(label, (2, 0, 1))

    print(image.shape)

    for i in range(classes):
        class_stats[i] += np.sum(label == i)
        total += np.sum(label == i)

    # image = np.clip(image, min_HU, max_HU)

    image = (image - image.min())/(image.max()-image.min())

    reshaped_image = reshape(image, (1, 1, 1), append_value=0, new_shape=(64, 64, 64))
    reshaped_label = reshape(label, (1, 1, 1), append_value=0, new_shape=(64, 64, 64))
    result = np.stack((reshaped_image, reshaped_label))

    np.save(os.path.join(output_dir, f.split('.')[0]+'.npy'), result)
    print(f)

print(total)
for i in range(classes):
    print(class_stats[i], class_stats[i]/total)
