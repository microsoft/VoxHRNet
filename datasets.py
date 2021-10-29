# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torch
import nibabel
import os
import numpy as np
import transforms


nibabel.Nifti1Header.quaternion_threshold = -1e-06


class MyCustomDataset(Dataset):

    def __init__(self, img_list, label_list, is_train, label_map_file_name = None):

        super(MyCustomDataset, self).__init__()
        self.img_list = img_list
        self.label_list = label_list
        self.is_train = is_train
        self.count = len(img_list)
        self.label_map = np.load(label_map_file_name, allow_pickle = True).item()

        return

    def reshape_img_dims(self, img):
        
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        else:
            img = img.permute(3, 0, 1, 2)
        
        return img

    def transform_volume(self, img, label, prob = 0.2):

        if np.random.rand() > 1.0:
            img, label = transforms.random_affine(img, label)
        if np.random.rand() > prob:
            img, label = transforms.random_elastic(img, label)
        if np.random.rand() > 1.0:
            img, label = transforms.block_random_regions(img, label)
        if np.random.rand() > prob:
            img = transforms.add_gaussian_noise(img)
        if np.random.rand() > 1.0:
            img, label = transforms.shift_scale(img, label)

        return img, label

    def remap_label(self, label):

        ret = np.zeros(label.shape, dtype = label.dtype)

        for to_label, from_label in self.label_map.items():
            ret[label == from_label] = to_label
        
        return ret

    def __getitem__(self, index):
        
        img = nibabel.load(self.img_list[index]).get_data()
        label = nibabel.load(self.label_list[index]).get_data()
        
        label = self.remap_label(label)
        
        img = (img - np.mean(img)) / np.std(img)
        if self.is_train:
            img, label = self.transform_volume(img, label)
        
        img = torch.Tensor(img)
        label = torch.LongTensor(label)
        
        img = self.reshape_img_dims(img)
        
        return (img, label)

    def __len__(self):

        return self.count


def read_img(input_path):
    
    img = nibabel.load(input_path)
    data = img.get_data()
    affine = img.affine

    return data, affine


def get_dataset_file_list(config, div_name):

    root_dir = os.path.join(config['DATASET']['ROOT'], config['DATASET']['DATASET'])


    group_names = config[div_name]['GROUP_NAME']
    group_cnts = config[div_name]['GROUP_CNT']
    
    imgs = []
    segs = []
    for group_name, group_cnt in zip(group_names, group_cnts):
        for ind in range(group_cnt):
            subject_id = '{}{:03d}'.format(group_name, ind + 1)
            img_file_name = 'orig_{}.nii.gz'.format(subject_id)
            seg_file_name = 'aseg_{}.nii.gz'.format(subject_id)
            img_file_path = os.path.join(root_dir, img_file_name)
            seg_file_path = os.path.join(root_dir, seg_file_name)
            imgs.append(img_file_path)
            segs.append(seg_file_path)
    
    return imgs, segs


def get_dataloader(config):

    train_bsize = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    val_bsize = config.VALIDATE.BATCH_SIZE_PER_GPU * len(config.GPUS)
    test_bsize = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    train_img_names, train_seg_names = get_dataset_file_list(config, div_name = 'TRAIN')
    val_img_names, val_seg_names = get_dataset_file_list(config, div_name = 'VALIDATE')
    test_img_names, test_seg_names = get_dataset_file_list(config, div_name = 'TEST')
    train_set = MyCustomDataset(train_img_names, train_seg_names, is_train = True, label_map_file_name = config.DATASET.DATASET_DICT)
    val_set = MyCustomDataset(val_img_names, val_seg_names, is_train = False, label_map_file_name = config.DATASET.DATASET_DICT)
    test_set = MyCustomDataset(test_img_names, test_seg_names, is_train = False, label_map_file_name = config.DATASET.DATASET_DICT)
    train_loader = DataLoader(train_set, batch_size = train_bsize, shuffle = True, pin_memory = True, num_workers = config.WORKERS)
    val_loader = DataLoader(val_set, batch_size = val_bsize, shuffle = False, pin_memory = True, num_workers = config.WORKERS)
    test_loader = DataLoader(test_set, batch_size = test_bsize, shuffle = False, pin_memory = True, num_workers = config.WORKERS)

    return train_loader, val_loader, test_loader
    
