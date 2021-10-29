# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config import update_config
from config import _C as config
from apex import amp

import argparse
import utils
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import voxhrnet
import voxresnet
import unet
import unetpp
import datasets
import numpy as np
import os
import nibabel
import modelsummary
import radam
import SimpleITK as sitk


def parse_args():

    parser = argparse.ArgumentParser(description = 'Train VoxHRNet')
    parser.add_argument('--cfg', help = 'configuration file', required = True, type = str)
    parser.add_argument('opts', help = 'modify config options using the command-line', default = None, nargs = argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


def test_model():

    dump_input = torch.rand((1, 1, 181, 217, 181))
    # model = voxhrnet.HighResolutionNet(config, in_channels = 1)
    # model = unet.UNet(1, 55, 8)
    # model = voxresnet.VoxResNet(1, 55, 16)
    model = unetpp.Nested_UNet(1, 55, 9)
    # model = nn.DataParallel(model, device_ids = (0,))
    summary_output = modelsummary.get_model_summary(model.cuda(), dump_input.cuda())
    # summary_output = modelsummary.get_model_summary(model, dump_input)
    print(summary_output)

    return 0


def transform_label(label):

    ret = torch.zeros(label.shape, dtype = label.dtype)
    label_map = np.load(config.DATASET.DATASET_DICT, allow_pickle = True).item()

    for to_label, from_label in label_map.items():
        ret[label == from_label] = to_label

    return ret


def evaluate_one_pair(seg_path0, seg_path1, num_classes):

    seg0, _ = datasets.read_img(seg_path0)
    seg1, _ = datasets.read_img(seg_path1)
    seg0 = torch.LongTensor(seg0)
    seg1 = torch.LongTensor(seg1)
    seg0 = transform_label(seg0)
    seg1 = transform_label(seg1)
    seg0 = seg0.cuda()
    seg1 = seg1.cuda()

    dice_all = np.zeros(num_classes)
    for ci in range(num_classes):
        m0 = (seg0 == ci)
        m1 = (seg1 == ci)
        dice = (2 * (m0 * m1).sum(dim = (0, 1, 2)).float() / ((m0 + m1).sum(dim = (0, 1, 2))).float()).cpu().numpy()
        dice_all[ci] = dice

    return dice_all


def evaluate_saved_model_from_loader(config):

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(ele) for ele in config.GPUS])
    gpus = list(range(len(config.GPUS)))
    num_classes = config.DATASET.NUM_CLASSES

    assert(config.MODEL.NAME in ['voxhrnet', 'unet', 'voxresnet', 'unet++'])
    if config.MODEL.NAME == 'voxhrnet':
        model = voxhrnet.HighResolutionNet(config, in_channels = 1).cuda()
    elif config.MODEL.NAME == 'unet':
        model = unet.UNet(in_dim = 1, out_dim = num_classes, num_filters = 4).cuda()
    elif config.MODEL.NAME == 'voxresnet':
        model = voxresnet.VoxResNet(in_channels = 1, n_classes = num_classes, num_channels = 16).cuda()
    elif config.MODEL.NAME == 'unet++':
        model = unetpp.Nested_UNet(in_ch = 1, out_ch = num_classes, n_channels = 9).cuda()

    optimizer = radam.RAdam(model.parameters())
    if config.USE_AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level = config.AMP_OPT_LEVEL)
    model = nn.DataParallel(model, device_ids = gpus).cuda()
    checkpoint = torch.load(config.TEST.TEST_STATE_PATH, map_location = torch.device('cpu'))
    model.module.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if config.USE_AMP:
        amp.load_state_dict(checkpoint['amp'])
    
    _, _, test_loader = datasets.get_dataloader(config)
    
    cur_dice, cur_hd, cur_test_time = utils.compute_metrics(model, test_loader, num_classes, compute_hd = True)
    
    np.set_printoptions(precision = 4, floatmode = 'fixed', suppress = True)
    
    np.save('dice.npy', cur_dice)
    np.save('hd.npy', cur_hd)
    
    print('Dice, Subject-wise:')
    print(cur_dice[:, 1:].mean(1))
    print('Dice, Class-wise:')
    print(cur_dice.mean(0))
    print('HD, Subject-wise:')
    print(cur_hd[:, 1:].mean(1))
    print('HD, Class-wise:')
    print(cur_hd.mean(0))
    print('\n')
    
    print('Dice, Mean:')
    print(cur_dice[:, 1:].mean())
    print('Dice, Mean Square:')
    print(np.square(cur_dice[:, 1:]).mean())
    print('\n')
    
    print('HD, Mean:')
    print(cur_hd[:, 1:].mean())
    print('HD, Mean Square:')
    print(np.square(cur_hd[:, 1:]).mean())
    print('\n')
    
    print('Test Time, Mean:')
    print(cur_test_time.mean())
    print('Test Time, Mean Square:')
    print(np.square(cur_test_time).mean())
    print('\n')
    
    return 0


def evaluate_from_path(compute_hd = True):

    input_dir = '../data/Hammers_test'
    num_classes = 96
    group_names = ['TEST']
    group_cnts = [10]
    
    dice_all = np.zeros((sum(group_cnts), num_classes))
    hd_all = np.zeros((sum(group_cnts), num_classes))
    hd_computer = sitk.HausdorffDistanceImageFilter()
    toti = 0
    for group_name, group_cnt in zip(group_names, group_cnts):
        for ind in range(group_cnt):

            subject_id = '{}{:03d}'.format(group_name, ind + 1)

            seg_file_name = 'aseg_{}.nii.gz'.format(subject_id)
            seg_path = os.path.join(input_dir, seg_file_name)
            pred_seg_name = 'joint_{}_.nii.gz'.format(subject_id)
            pred_seg_path = os.path.join(input_dir, pred_seg_name)
            seg0, _ = datasets.read_img(seg_path)
            seg1, _ = datasets.read_img(pred_seg_path)

            seg0 = torch.LongTensor(seg0)
            seg1 = torch.LongTensor(seg1)
            seg0 = transform_label(seg0)
            seg1 = transform_label(seg1)
            seg0 = seg0.cuda()
            seg1 = seg1.cuda()
            
            for ci in range(num_classes):
                m0 = (seg0 == ci).int()
                m1 = (seg1 == ci).int()
                dice_all[toti, ci] = (2 * (m0 * m1).sum(dim = (0, 1, 2)).float() / ((m0 + m1).sum(dim = (0, 1, 2))).float()).cpu().numpy()
                
                if compute_hd:
                    m0_cnt = m0.sum()
                    m1_cnt = m1.sum()
                    m0 = m0.cpu().numpy()
                    m1 = m1.cpu().numpy()
                    
                    if m0_cnt == 0 or m1_cnt == 0:
                        hd_all[toti, ci] = np.float('inf')
                    else:
                        hd_computer.Execute(sitk.GetImageFromArray(m0), sitk.GetImageFromArray(m1))
                        hd_all[toti, ci] = hd_computer.GetHausdorffDistance()

            toti += 1

    np.set_printoptions(precision = 4, floatmode = 'fixed', suppress = True)
    
    print('Dice, Subject-wise:')
    print(dice_all[:, 1:].mean(1))
    print('Dice, Class-wise:')
    print(dice_all.mean(0))
    print('HD, Subject-wise:')
    print(hd_all[:, 1:].mean(1))
    print('HD, Class-wise:')
    print(hd_all.mean(0))
    print('\n')
    
    print('Dice, Mean:')
    print(dice_all[:, 1:].mean())
    print('Dice, Mean Square:')
    print(np.square(dice_all[:, 1:]).mean())
    print('\n')
    
    print('HD, Mean:')
    print(hd_all[:, 1:].mean())
    print('HD, Mean Square:')
    print(np.square(hd_all[:, 1:]).mean())
    print('\n')

    return 0


def predict_saved_model_from_path(config):

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(ele) for ele in config.GPUS])
    gpus = list(range(len(config.GPUS)))
    num_classes = config.DATASET.NUM_CLASSES

    assert(config.MODEL.NAME in ['voxhrnet', 'unet', 'voxresnet', 'unet++'])
    if config.MODEL.NAME == 'voxhrnet':
        model = voxhrnet.HighResolutionNet(config, in_channels = 1).cuda()
    elif config.MODEL.NAME == 'unet':
        model = unet.UNet(in_dim = 1, out_dim = num_classes, num_filters = 4).cuda()
    elif config.MODEL.NAME == 'voxresnet':
        model = voxresnet.VoxResNet(in_channels = 1, n_classes = num_classes, num_channels = 16).cuda()
    elif config.MODEL.NAME == 'unet++':
        model = unetpp.Nested_UNet(in_ch = 1, out_ch = num_classes, n_channels = 9).cuda()

    optimizer = radam.RAdam(model.parameters())
    if config.USE_AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level = config.AMP_OPT_LEVEL)
    model = nn.DataParallel(model, device_ids = gpus).cuda()
    checkpoint = torch.load(config.TEST.TEST_STATE_PATH, map_location = torch.device('cpu'))
    model.module.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if config.USE_AMP:
        amp.load_state_dict(checkpoint['amp'])
    
    root_dir = os.path.join(config.DATASET.ROOT, config.DATASET.DATASET)
    pred_dir = os.path.join('output')

    os.makedirs(pred_dir, exist_ok = True)

    group_names = config.TEST.GROUP_NAME
    group_cnts = config.TEST.GROUP_CNT
    
    label_map = np.load(config.DATASET.DATASET_DICT, allow_pickle = True).item()

    for group_name, group_cnt in zip(group_names, group_cnts):
        for ind in range(group_cnt):

            subject_id = '{}{:03d}'.format(group_name, ind + 1)

            img_file_name = 'orig_{}.nii.gz'.format(subject_id)
            img_path = os.path.join(root_dir, img_file_name)
            seg_file_name = 'aseg_{}.nii.gz'.format(subject_id)
            seg_path = os.path.join(root_dir, seg_file_name)
            pred_seg_name = 'aseg_{}_pred.nii.gz'.format(subject_id)
            pred_seg_path = os.path.join(pred_dir, pred_seg_name)
            false_pred_name = 'aseg_{}_false_pred.nii.gz'.format(subject_id)
            false_pred_path = os.path.join(pred_dir, false_pred_name)
            max_prob_name = 'aseg_{}_pmap.nii.gz'.format(subject_id)
            max_prob_path = os.path.join(pred_dir, max_prob_name)
            img, affine = datasets.read_img(img_path)
            seg, affine = datasets.read_img(seg_path)
            seg = torch.LongTensor(seg)
            
            with torch.no_grad():
                img = torch.Tensor(img).unsqueeze(0).unsqueeze(1).cuda()
                max_prob, pred = F.softmax(model(img), dim = 1).squeeze().max(0)
                mapped_pred = torch.zeros(pred.shape, dtype = pred.dtype).cuda()
                mapped_seg = torch.zeros(seg.shape, dtype = seg.dtype).cuda()
                for from_label, to_label in label_map.items():
                    mapped_pred[pred == from_label] = int(to_label)
                    mapped_seg[seg == to_label] = int(to_label)
                max_prob *= (mapped_pred > 0).float()
                false_pred = (mapped_pred != mapped_seg)
                data = np.int16(mapped_pred.cpu().numpy())
                nib = nibabel.Nifti1Image(data, affine)
                nibabel.save(nib, pred_seg_path)
                data = np.int16(false_pred.cpu().numpy())
                nib = nibabel.Nifti1Image(data, affine)
                nibabel.save(nib, false_pred_path)
                data = np.float32(max_prob.cpu().numpy())
                nib = nibabel.Nifti1Image(data, affine)
                nibabel.save(nib, max_prob_path)

    return 0


if __name__ == '__main__':

    args = parse_args()
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # test_model()
    evaluate_saved_model_from_loader(config)
    
