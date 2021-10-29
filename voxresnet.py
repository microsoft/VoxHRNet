# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, downsample = None):

        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.net = nn.Sequential(
                    nn.Conv3d(in_planes, planes, 1, 1, 0, bias = False),
                    nn.BatchNorm3d(planes),
                    nn.ReLU(),
                    nn.Conv3d(planes, planes, 3, 1, 1, bias = False),
                    nn.BatchNorm3d(planes),
                    nn.ReLU(),
                    nn.Conv3d(planes, planes * self.expansion, 1, 1, 0, bias = False),
                    nn.BatchNorm3d(planes * self.expansion)
                )
        self.relu = nn.ReLU()

        return

    def forward(self, x):

        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)

        y = self.relu(self.net(x) + residual)

        return y


class VoxResModule(nn.Module):

    def __init__(self, num_channels):

        super(VoxResModule, self).__init__()
        self.net = nn.Sequential(
                    nn.BatchNorm3d(num_channels),
                    nn.ReLU(),
                    nn.Conv3d(num_channels, num_channels, 3, 1, 1),
                    nn.BatchNorm3d(num_channels),
                    nn.ReLU(),
                    nn.Conv3d(num_channels, num_channels, 3, 1, 1)
                )

        return

    def forward(self, x):

        return x + self.net(x)


class VoxResNet(nn.Module):

    def __init__(self, in_channels, n_classes, num_channels = 32):

        super(VoxResNet, self).__init__()
        self.stem_net = nn.Sequential(
                    nn.Conv3d(in_channels, 32, 3, 2, 1, bias = False),
                    nn.BatchNorm3d(32),
                    nn.ReLU(),
                    self._make_layer(Bottleneck, 32, 16, 2)
                )
        self.net_h0 = nn.Sequential(
                    nn.Conv3d(64, num_channels, 3, 1, 1),
                    nn.BatchNorm3d(num_channels),
                    nn.ReLU(),
                    nn.Conv3d(num_channels, num_channels, 3, 1, 1)
                )
        self.net_c0 = nn.Sequential(
                    nn.ConvTranspose3d(num_channels, num_channels, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv3d(num_channels, n_classes, 3, 1, 1)
                )
        self.net_h1 = nn.Sequential(
                    nn.BatchNorm3d(num_channels),
                    nn.ReLU(),
                    nn.Conv3d(num_channels, num_channels * 2, 3, 2, 1),
                    VoxResModule(num_channels = num_channels * 2),
                    VoxResModule(num_channels = num_channels * 2)
                )
        self.net_c1 = nn.Sequential(
                    nn.ConvTranspose3d(num_channels * 2, num_channels * 2, 4, 2, 1),
                    nn.ReLU(),
                    nn.Conv3d(num_channels * 2, n_classes, 3, 1, 1)
                )
        self.net_h2 = nn.Sequential(
                    nn.BatchNorm3d(num_channels * 2),
                    nn.ReLU(),
                    nn.Conv3d(num_channels * 2, num_channels * 2, 3, 2, 1),
                    VoxResModule(num_channels = num_channels * 2),
                    VoxResModule(num_channels = num_channels * 2)
                )
        self.net_c2 = nn.Sequential(
                    nn.ConvTranspose3d(num_channels * 2, num_channels * 2, 6, 4, 1),
                    nn.ReLU(),
                    nn.Conv3d(num_channels * 2, n_classes, 3, 1, 1)
                )
        self.net_h3 = nn.Sequential(
                    nn.BatchNorm3d(num_channels * 2),
                    nn.ReLU(),
                    nn.Conv3d(num_channels * 2, num_channels * 2, 3, 2, 1),
                    VoxResModule(num_channels = num_channels * 2),
                    VoxResModule(num_channels = num_channels * 2)
                )
        self.net_c3 = nn.Sequential(
                    nn.ConvTranspose3d(num_channels * 2, num_channels * 2, 10, 8, 1),
                    nn.ReLU(),
                    nn.Conv3d(num_channels * 2, n_classes, 3, 1, 1)
                )
        self.softmax = nn.Softmax(dim = 1)

        return

    def _make_layer(self, block, in_planes, planes, num_block):

        if in_planes == planes * block.expansion:
            downsample = None
        else:
            downsample = nn.Sequential(
                        nn.Conv3d(in_planes, planes * block.expansion, 1, 1, 0, bias = False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )
        layers = [block(in_planes, planes, downsample)]
        layers.extend([block(planes * block.expansion, planes) for i in range(num_block - 1)])

        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        original_shape = x.shape
        x = self.stem_net(x)
        h = self.net_h0(x)
        c0 = self.net_c0(h)
        h = self.net_h1(h)
        c1 = self.net_c1(h)
        h = self.net_h2(h)
        c2 = self.net_c2(h)
        h = self.net_h3(h)
        c3 = self.net_c3(h)
        c1 = F.interpolate(c1, size = c0.shape[-3:], mode = 'trilinear', align_corners = False)
        c2 = F.interpolate(c2, size = c0.shape[-3:], mode = 'trilinear', align_corners = False)
        c3 = F.interpolate(c3, size = c0.shape[-3:], mode = 'trilinear', align_corners = False)
        c = c0 + c1 + c2 + c3
        c = F.interpolate(c, size = original_shape[-3:], mode = 'trilinear', align_corners = False)

        return c

