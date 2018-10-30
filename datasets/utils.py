import numpy as np


def reshape(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0]+x_offset, y_offset:orig_img.shape[1]+y_offset, z_offset:orig_img.shape[2]+z_offset] = orig_img
    # insert temp_img.min() as background value

    return reshaped_image
