# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yacs.config import CfgNode as CN


_C = CN()
_C.OUTPUT_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.USE_AMP = False
_C.AMP_OPT_LEVEL = 'O1'
_C.LOSS = 'ce'

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.MODEL = CN()
_C.MODEL.NAME = 'seg_voxhrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.EXTRA = CN(new_allowed = True)

_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = ''
_C.DATASET.DATASET_DICT = ''
_C.DATASET.NUM_CLASSES = 0

_C.TRAIN = CN()
_C.TRAIN.GROUP_NAME = ['TRAIN']
_C.TRAIN.GROUP_CNT = [15]
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.RESUME = False
_C.TRAIN.RESUME_STATE_PATH = ''
_C.TRAIN.SAVE_DIR = ''
_C.TRAIN.SAVE_FREQ = 100
_C.TRAIN.PRINT_FREQ = 10

_C.VALIDATE= CN()
_C.VALIDATE.GROUP_NAME = ['VALIDATE']
_C.VALIDATE.GROUP_CNT = [5]
_C.VALIDATE.BATCH_SIZE_PER_GPU = 32

_C.TEST = CN()
_C.TEST.GROUP_NAME = ['TEST']
_C.TEST.GROUP_CNT = [20]
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.TEST_STATE_PATH = ''


def update_config(cfg, args):
    
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return

