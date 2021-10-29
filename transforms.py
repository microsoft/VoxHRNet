# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom, affine_transform
from scipy.stats import mode
from skimage import exposure

import math
import numpy as np
import nibabel
import warnings


warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def random_crop(img, label = None, crop_size = 128):
    
    tile = []
    for cur_dim in img.shape:
        pos = np.random.randint(0, max(0, cur_dim - crop_size))
        tile.append(slice(pos, min(cur_dim, pos + crop_size)))
    tile = tuple(tile)
    
    if label is None:
        return img[tile]
    else:
        return img[tile], label[tile]


def random_gamma(img):
    
    img = (img - img.min()) / (img.max() - img.min())
    
    return exposure.adjust_gamma(img, gamma = np.random.rand() * 0.5 + 1.0)


def invert(img):
    
    return img.max() - img


def clahe(img):
    
    return exposure.equalize_hist(img)


def add_gaussian_blur(img):
    
    return gaussian_filter(img, sigma = np.random.rand() * 2)


def add_gaussian_offset(img):
    
    offset = np.random.rand() * 0.05

    return img + offset


def add_gaussian_noise(img):
    
    mean = 0
    sigma = np.random.rand() * 0.1
    
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)

    return img + gauss


def block_random_regions(img, label = None):
    
    alpha = 0.5
    beta = 0.75
    block_shape = (np.random.rand(3) * alpha * np.array(img.shape)).astype(np.int)
    block_origin = (np.random.rand(3) * beta * np.array(img.shape)).astype(np.int)
    bg_label = 0
    bg_intensity = mode(img[label == bg_label]).mode[0]
    
    if label is None:
        img[block_origin[0] : block_origin[0] + block_shape[0], block_origin[1] : block_origin[1] + block_shape[1], block_origin[2] : block_origin[2] + block_shape[2]] = bg_intensity
        return img
    else:
        img[block_origin[0] : block_origin[0] + block_shape[0], block_origin[1] : block_origin[1] + block_shape[1], block_origin[2] : block_origin[2] + block_shape[2]] = bg_intensity
        label[block_origin[0] : block_origin[0] + block_shape[0], block_origin[1] : block_origin[1] + block_shape[1], block_origin[2] : block_origin[2] + block_shape[2]] = bg_label
        return img, label


def random_affine(img, label = None):
    
    rotation_theta = np.random.rand(3) * 20 / 180 * math.pi
    rotation_tform0 = np.array([[1, 0, 0, 0],
                                [0, np.cos(rotation_theta[0]), -np.sin(rotation_theta[0]), 0],
                                [0, np.sin(rotation_theta[0]), np.cos(rotation_theta[0]), 0],
                                [0, 0, 0, 1]])
    rotation_tform1 = np.array([[np.cos(rotation_theta[1]), 0, np.sin(rotation_theta[1]), 0],
                                [0, 1, 0, 0],
                                [-np.sin(rotation_theta[1]), 0, np.cos(rotation_theta[1]), 0],
                                [0, 0, 0, 1]])
    rotation_tform2 = np.array([[np.cos(rotation_theta[2]), -np.sin(rotation_theta[2]), 0, 0],
                                [np.sin(rotation_theta[2]), np.cos(rotation_theta[2]), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
                               
    translation_offset = np.array(img.shape) * (np.random.rand(3) * 0.05)
    translation_tform = np.array([[1, 0, 0, translation_offset[0]],
                                  [0, 1, 0, translation_offset[1]],
                                  [0, 0, 1, translation_offset[2]],
                                  [0, 0, 0, 1]])
    
    shear_theta = np.random.rand(6) * 6 / 180 * math.pi
    shear_tform = np.array([[1, np.tan(shear_theta[0]), np.tan(shear_theta[1]), 0],
                            [np.tan(shear_theta[2]), 1, np.tan(shear_theta[3]), 0],
                            [np.tan(shear_theta[4]), np.tan(shear_theta[5]), 1, 0],
                            [0, 0, 0, 1]])
    
    zoom_factor = np.random.rand(3) * 0.2 + 1.0
    zoom_tform = np.array([[zoom_factor[0], 0, 0, 0],
                           [0, zoom_factor[1], 0, 0],
                           [0, 0, zoom_factor[2], 0],
                           [0, 0, 0, 1]])
    
    tform = rotation_tform0.dot(rotation_tform1).dot(rotation_tform2).dot(translation_tform).dot(shear_tform).dot(zoom_tform)
    if label is None:
        return affine_transform(img, tform, order = 0, mode = 'nearest')
    else:
        return affine_transform(img, tform, order = 0, mode = 'nearest'), affine_transform(label, tform, order = 0, mode = 'nearest')
    

def random_elastic(img, label = None):

    alpha = np.array(img.shape) * 2
    sigma = np.array(img.shape) * (np.random.rand(3) * 0.02 + 0.04)

    dx = gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma[0]) * alpha[0]
    dy = gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma[1]) * alpha[1]
    dz = gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma[2]) * alpha[2]

    x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]), indexing = 'ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))]

    if label is None:
        return map_coordinates(img, indices, order = 0, mode = 'nearest').reshape(img.shape)
    else:
        return map_coordinates(img, indices, order = 0, mode = 'nearest').reshape(img.shape), map_coordinates(label, indices, order = 0, mode = 'nearest').reshape(label.shape)


def shift_scale(img, label = None):
    
    min_delta, max_delta = -5, 5
    
    ori_shape = img.shape
    deltas = [np.random.randint(min_delta, max_delta + 1) for _ in img.shape]
    pad_widths = []
    slices = []
    for cur_dt, cur_dim in zip(deltas, ori_shape):
        pad_widths.append((max(0, cur_dt), max(0, cur_dt)))
        cur_dim += max(0, cur_dt * 2)
        slices.append(slice(max(0, 0 - cur_dt), min(cur_dim, cur_dim + cur_dt)))
    pad_widths = tuple(pad_widths)
    slices = tuple(slices)
    
    img = np.pad(img, pad_widths, mode = 'edge')[slices]
    scales = [to_dim / from_dim for from_dim, to_dim in zip(img.shape, ori_shape)]
    img = zoom(img, scales, order = 1)
    
    if label is None:
        return img
    else:
        label = np.pad(label, pad_widths, mode = 'constant', constant_values = 0)[slices]
        label = zoom(label, scales, output = np.int32, order = 0)

        return img, label

def resize_label(label, scale_factor = 0.5):

    return zoom(label, scale_factor, output = np.int32, order = 0)
    

if __name__ == '__main__':

    input_img_path = '../data/TRANSFORM_TEST/orig_TEST007.nii.gz'
    input_seg_path = '../data/TRANSFORM_TEST/aseg_TEST007.nii.gz'
    output_img_path = '../data/TRANSFORM_TEST/orig_elastic.nii.gz'
    output_seg_path = '../data/TRANSFORM_TEST/aseg_elastic.nii.gz'

    img = nibabel.load(input_img_path)
    img_data, img_affine = img.get_data(), img.affine
    seg = nibabel.load(input_seg_path)
    seg_data, seg_affine = seg.get_data(), seg.affine
    
    img_data = (img_data - np.mean(img_data)) / np.std(img_data)
    
    img_data, seg_data = random_affine(img_data, seg_data)
    # img_data, seg_data = random_elastic(img_data, seg_data)
    # img_data, seg_data = block_random_regions(img_data, seg_data)
    # img_data, seg_data = shift_scale(img_data, seg_data)
    # img_data = random_gamma(img_data)
    # img_data = clahe(img_data)
    # img_data = invert(img_data)
    # img_data = add_gaussian_noise(img_data)
    
    # img_data = (img_data - np.mean(img_data)) / np.std(img_data)
    img = nibabel.Nifti1Image(img_data, img_affine)
    nibabel.save(img, output_img_path)
    seg = nibabel.Nifti1Image(seg_data, seg_affine)
    nibabel.save(seg, output_seg_path)

