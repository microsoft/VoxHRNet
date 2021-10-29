# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.autograd import Variable, Function
from torch.nn.modules.loss import _Loss, _WeightedLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossEntropy(_WeightedLoss):

    def __init__(self, weight = None):

        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight = weight)

        return

    def forward(self, pred, target):

        loss = self.criterion(pred, target)

        return loss


class DiceLoss(_WeightedLoss):

    def __init__(self, eps = 0.0001):

        super(DiceLoss, self).__init__()
        self.eps = eps
        self.softmax = nn.Softmax(dim = 1)

        return

    def forward(self, pred, target):

        pred = self.softmax(pred)
        target = (pred.detach() * 0).scatter_(1, target.unsqueeze(1), 1)

        numerator = 2 * (pred * target).sum(dim = (2, 3, 4)) + self.eps
        denominator = (pred + target).sum(dim = (2, 3, 4)) + self.eps
        loss_per_channel = (1 - numerator / denominator)
        loss = loss_per_channel.sum() / pred.shape[0] / pred.shape[1]

        return loss


class GeneralizedDiceLoss(_WeightedLoss):

    def __init__(self, smooth = 0.0001):

        super(GeneralizedDiceLoss, self).__init__()
        self.smooth = smooth
        self.softmax = nn.Softmax(dim = 1)

        return

    def forward(self, pred, target):

        pred = self.softmax(pred)
        target = (pred.detach() * 0).scatter_(1, target.unsqueeze(1), 1)

        class_weights = Variable(1.0 / (target * target).sum(dim = (2, 3, 4)), requires_grad = False)

        numerator = ((pred * target).sum(dim = (2, 3, 4)) * class_weights).sum() + self.smooth
        denominator = ((pred + target).sum(dim = (2, 3, 4)) * class_weights).sum() + self.smooth

        loss = 1.0 - 2.0 * numerator / denominator

        return loss


class ActiveContourLoss(_WeightedLoss):

    def __init__(self):

        super(ActiveContourLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.epsilon = 1e-8
        self.w = 1
        self.C_1 = 1
        self.C_2 = 0
        self.lambdaP = 1

        return

    def forward(self, pred, target):

        pred = self.softmax(pred)
        target = (pred.detach() * 0).scatter_(1, target.unsqueeze(1), 1)

        original_shape = target.shape

        x = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        z = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        delta_x = x[:, :, 1:, :-2, :-2] ** 2
        delta_y = y[:, :, :-2, 1:, :-2] ** 2
        delta_z = y[:, :, :-2, :-2, 1:] ** 2
        delta_u = (delta_x + delta_y + delta_z).abs()

        lenth = self.w * (delta_u + self.epsilon).sqrt().sum()
        
        region_in = (pred[:, 0, :, :, :] * ((target[:, 0, :, : ,:] - self.C_1) ** 2)).sum().abs()
        region_out = ((1 - pred[:, 0, :, :, :]) * ((target[:, 0, :, : ,:] - self.C_2) ** 2)).sum().abs()
        
        loss = lenth + lambdaP * (region_in + region_out)

        return loss


class CombinedLoss(_Loss):

    def __init__(self):

        super(CombinedLoss, self).__init__()
        self.loss0 = CrossEntropy()
        self.loss1 = DiceLoss()

        return 

    def forward(self, pred, target):

        loss = self.loss0(pred, target) + self.loss1(pred, target)

        return loss


