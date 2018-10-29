from batchgenerators.transforms import Compose, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor, ReshapeTransform


def get_transforms(mode="train", target_size=128, reshape_factor=4, number_of_slices=5):
    tranform_list = []

    if mode == "train":
        tranform_list = [RandomCropTransform(crop_size=(target_size, target_size)),
                         # ResizeTransform(target_size=(target_size, target_size), order=0),
                         # MirrorTransform(axes=(2,)),
                         # SpatialTransform(patch_size=(32, 32), random_crop=False,
                         #                   patch_center_dist_from_border=target_size // 2,
                         #                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(40., 60.),
                         #                    do_rotation=True,
                         #                    angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                         #                    scale=(0.9, 1.4),
                         #                    border_mode_data="nearest", border_mode_seg="nearest"),
                         ]

    elif mode == "val":
        tranform_list = [CenterCropTransform(crop_size=(target_size, target_size)),
                         ResizeTransform(target_size=(32, 32), order=1),
                         # BrightnessTransform(mu=0, sigma=0.2),
                         # BrightnessMultiplicativeTransform(multiplier_range=(0.95, 1.1)),
                         ]

    tranform_list.append(NumpyToTensor())

    return Compose(tranform_list)