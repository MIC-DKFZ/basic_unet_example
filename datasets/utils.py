import numpy as np
from SimpleITK import SimpleITK


def load_dcm_image(dcm_dir):
    reader = SimpleITK.ImageSeriesReader()
    filenames_dicom = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(filenames_dicom)

    image_dicom = reader.Execute()
    image = SimpleITK.GetArrayFromImage(image_dicom)

    spacing = image_dicom.GetSpacing()

    return image, spacing


def load_label(label_dir):
    label_file = label_dir + '_mask_JW.nrrd'

    # images are 512 by 512 pixels each
    image_nrrd = SimpleITK.ReadImage(label_file)
    label = SimpleITK.GetArrayFromImage(image_nrrd)

    return label


def reshape(orig_img, original_spacing, append_value=-1024, new_shape=(512, 512, 512)):
    orig_img = SimpleITK.GetImageFromArray(orig_img)

    original_size = orig_img.GetSize()

    orig_img.SetSpacing(original_spacing)
    new_spacing = [1, 1, original_spacing[2]]  # image_dicom.GetSpacing()
    new_size = [int(round(original_size[0]) * (original_spacing[0]/new_spacing[0])),
                int(round(original_size[1]) * (original_spacing[1]/new_spacing[1])),
                int(round(original_size[2]) * (original_spacing[2]/new_spacing[2]))]

    orig_img = SimpleITK.Resample(orig_img, new_size, SimpleITK.Transform(),
                                  SimpleITK.sitkNearestNeighbor, orig_img.GetOrigin(),
                                  new_spacing, orig_img.GetDirection(), 0.0,
                                  orig_img.GetPixelID())

    temp_img = SimpleITK.GetArrayFromImage(orig_img)

    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = (new_shape[1] - temp_img.shape[1]) // 2
    z_offset = (new_shape[2] - temp_img.shape[2]) // 2

    reshaped_image[x_offset:temp_img.shape[0]+x_offset, y_offset:temp_img.shape[1]+y_offset, z_offset:temp_img.shape[2]+z_offset] = temp_img
    # insert temp_img.min() as background value

    return reshaped_image
