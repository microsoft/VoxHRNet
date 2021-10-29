# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import SimpleITK as sitk
import torch
import torch.nn as nn
import numpy as np
import time


def get_gpu_allocated(gpus = []):

    if len(gpus) == 0:
        ret = torch.cuda.max_memory_allocated()
    else:
        ret = sum([torch.cuda.max_memory_allocated(device = gpu) for gpu in gpus])
    
    return ret


def print_dice_info(cur_train_dice, cur_val_dice, best_val_dice):
    
    np.set_printoptions(precision = 4, floatmode = 'fixed', suppress = True)
    
    print('\n')
    print('Current training dice:')
    print('Subject-wise:')
    print(cur_train_dice[:, 1:].mean(1))
    print('Class-wise:')
    print(cur_train_dice.mean(0))
    print('Overall:')
    print(cur_train_dice[:, 1:].mean())
    print()
    
    print('Current validation dice:')
    print('Subject-wise:')
    print(cur_val_dice[:, 1:].mean(1))
    print('Class-wise:')
    print(cur_val_dice.mean(0))
    print('Overall:')
    print(cur_val_dice[:, 1:].mean())
    print()
    
    print('Best validation dice:')
    print(best_val_dice)
    print('\n')
    
    return 0


def compute_metrics(model, data_loader, num_classes, compute_hd = False):
    
    dice_all = np.zeros((len(data_loader.dataset), num_classes))
    hd_all = np.zeros((len(data_loader.dataset), num_classes))
    pred_time = np.zeros(len(data_loader.dataset))
    hd_computer = sitk.HausdorffDistanceImageFilter()

    last_batch_tail = 0

    for batch_idx, sample in enumerate(data_loader):

        img, label = sample
        batch_tail = last_batch_tail + label.shape[0]

        with torch.no_grad():
            
            img = img.cuda()
            label = label.cuda()
            
            time_before_pass = time.time()
            
            pred = model(img)
            
            time_after_pass = time.time()
            pred_time[batch_idx] = time_after_pass - time_before_pass
            
            pred = pred.argmax(1)

        for di in range(num_classes):
            m0 = (pred == di).int()
            m1 = (label == di).int()
            dice = (2 * (m0 * m1).sum(dim = (1, 2, 3)).float() / ((m0 + m1).sum(dim = (1, 2, 3))).float()).cpu().numpy()
            dice_all[last_batch_tail:batch_tail, di] = dice
            if compute_hd:
                m0_cnt = m0.sum(dim = (1, 2, 3))
                m1_cnt = m1.sum(dim = (1, 2, 3))
                m0 = m0.cpu().numpy()
                m1 = m1.cpu().numpy()
                for bi in range(last_batch_tail, batch_tail):
                    if m0_cnt[bi - last_batch_tail] == 0 or m1_cnt[bi - last_batch_tail] == 0:
                        hd_all[bi, di] = np.float('inf')
                    else:
                        hd_computer.Execute(sitk.GetImageFromArray(m0[bi - last_batch_tail]), sitk.GetImageFromArray(m1[bi - last_batch_tail]))
                        hd_all[bi, di] = hd_computer.GetHausdorffDistance()
        
        last_batch_tail += label.shape[0]
        
        torch.cuda.empty_cache()
    
    if compute_hd:
        return dice_all, hd_all, pred_time
    else:
        return dice_all, pred_time

