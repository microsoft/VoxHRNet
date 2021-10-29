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
import losses
import time
import os
import radam


def parse_args():

    parser = argparse.ArgumentParser(description = 'Train VoxHRNet')
    parser.add_argument('--cfg', help = 'configuration file', required = True, type = str)
    parser.add_argument('opts', help = 'modify config options using the command-line', default = None, nargs = argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


if __name__ == '__main__':

    args = parse_args()

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(ele) for ele in config.GPUS])
    gpus = list(range(len(config.GPUS)))
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    num_epochs = config.TRAIN.END_EPOCH
    num_classes = config.DATASET.NUM_CLASSES
    output_dir = config.TRAIN.SAVE_DIR
    save_frequency = config.TRAIN.SAVE_FREQ
    print_frequency = config.TRAIN.PRINT_FREQ
    
    os.makedirs(output_dir, exist_ok = True)

    assert(config.LOSS in ['ce', 'dice', 'comb'])
    if config.LOSS == 'ce':
        criterion = losses.CrossEntropy()
    elif config.LOSS == 'dice':
        criterion = losses.DiceLoss()
    elif config.LOSS == 'comb':
        criterion = losses.CombinedLoss()
    
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
    
    train_loader, val_loader, _ = datasets.get_dataloader(config)
    
    max_gpu_alloc = 0
    best_val_dice = 0
    if config.TRAIN.RESUME:
        checkpoint = torch.load(config.TRAIN.RESUME_STATE_PATH, map_location = torch.device('cpu'))
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        begin_epoch = checkpoint['epoch']
        best_val_dice = checkpoint['best_val_dice']
        if config.USE_AMP:
            amp.load_state_dict(checkpoint['amp'])
        del checkpoint

    for epoch in range(begin_epoch, num_epochs):
        
        print('Epoch {}:\nLoss:'.format(epoch), end = '')
        cur_epoch_loss = np.zeros(len(train_loader.dataset))
        time_before_epoch = time.time()
        
        for batch_idx, sample in enumerate(train_loader):

            img, label = sample

            img = img.cuda()
            label = label.cuda()
            
            pred = model(img)
            loss = criterion(pred, label)
            
            cur_epoch_loss[batch_idx] = loss.item()
            print(' {:0.4f}'.format(loss.item()), end = '', flush = True)

            optimizer.zero_grad()
            if config.USE_AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            optimizer.step()
            
            torch.cuda.empty_cache()
        
        print()
        epoch_elapsed_time = time.time() - time_before_epoch
        
        cur_val_dice, _ = utils.compute_metrics(model, val_loader, num_classes)
        max_gpu_alloc = max(max_gpu_alloc, utils.get_gpu_allocated(gpus))
        
        if cur_val_dice[:, 1:].mean() > best_val_dice:
            best_val_dice = cur_val_dice[:, 1:].mean()
            checkpoint = {'state_dict' : model.module.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : epoch + 1, 'best_val_dice' : best_val_dice}
            if config.USE_AMP:
                checkpoint['amp'] = amp.state_dict()
            torch.save(checkpoint, os.path.join(output_dir, 'best.pth.tar'))
            del checkpoint
        
        if epoch % save_frequency == 0:
            checkpoint = {'state_dict' : model.module.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : epoch + 1, 'best_val_dice' : best_val_dice}
            if config.USE_AMP:
                checkpoint['amp'] = amp.state_dict()
            torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.pth.tar'))
            del checkpoint

        if epoch % print_frequency == 0:
            cur_train_dice, _ = utils.compute_metrics(model, train_loader, num_classes)
            utils.print_dice_info(cur_train_dice, cur_val_dice, best_val_dice)

