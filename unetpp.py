# --------------------------------------------------------------------------
# Source: https://gist.github.com/jinglescode/9d9ed6027e62e389e3165b59209e838e
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


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class Nested_UNet(nn.Module):

    def __init__(self, in_ch, out_ch, n_channels = 64):
        super(Nested_UNet, self).__init__()

        n1 = n_channels
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.stem_net = nn.Sequential(
                    nn.Conv3d(in_ch, 32, 3, 2, 1, bias = False),
                    nn.BatchNorm3d(32),
                    nn.ReLU(),
                    self._make_layer(Bottleneck, 32, 16, 2)
                )


        self.conv0_0 = conv_block_nested(64, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)

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

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = F.interpolate(x1_0, size = x0_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x0_1 = self.conv0_1(torch.cat([x0_0, x1_0], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = F.interpolate(x2_0, size = x1_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x1_1 = self.conv1_1(torch.cat([x1_0, x2_0], 1))
        x1_1 = F.interpolate(x1_1, size = x0_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = F.interpolate(x3_0, size = x2_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_0], 1))
        x2_1 = F.interpolate(x2_1, size = x1_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1], 1))
        x1_2 = F.interpolate(x1_2, size = x0_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = F.interpolate(x4_0, size = x3_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0], 1))
        x3_1 = F.interpolate(x3_1, size = x2_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1], 1))
        x2_2 = F.interpolate(x2_2, size = x1_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2], 1))
        x1_3 = F.interpolate(x1_3, size = x0_0.shape[-3:], mode = 'trilinear', align_corners = True)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3], 1))

        output = self.final(x0_4)

        output = F.interpolate(output, size = original_shape[-3:], mode = 'trilinear', align_corners = True)

        return output
