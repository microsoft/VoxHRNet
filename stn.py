# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Source: https://github.com/GuangYang98/ISTN-Custom/blob/9a1fb7163b5d180a5cb26f9bb835383261ac92fc/pymira_custom/nets/stn.py
# Modified by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BSplineSTN3D(nn.Module):

    def __init__(self, input_channels, device, input_size = (90, 90, 90), control_point_spacing=(10, 10, 10)):
        
        super(BSplineSTN3D, self).__init__()
        # Cuda params
        self.device = device
        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float

        self.input_size = input_size
        self.control_point_spacing = np.array(control_point_spacing)
        self.stride = self.control_point_spacing.astype(dtype=int).tolist()

        area = self.control_point_spacing[0] * self.control_point_spacing[1] * self.control_point_spacing[2]
        self.area = area.astype(float)
        cp_grid_shape = np.ceil(np.divide(self.input_size, self.control_point_spacing)).astype(dtype=int)

        # new image size after convolution
        self.inner_image_size = np.multiply(self.control_point_spacing, cp_grid_shape) - (
                self.control_point_spacing - 1)

        # add one control point at each side
        cp_grid_shape = cp_grid_shape + 2

        # image size with additional control points
        self.new_image_size = np.multiply(self.control_point_spacing, cp_grid_shape) - (self.control_point_spacing - 1)

        # center image between control points
        image_size_diff = self.inner_image_size - self.input_size
        image_size_diff_floor = np.floor((np.abs(image_size_diff) / 2)) * np.sign(image_size_diff)
        crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2) * np.sign(image_size_diff)
        self.crop_start = crop_start.astype(dtype=int)
        self.crop_end = image_size_diff_floor.astype(dtype=int)

        self.cp_grid_shape = [3] + cp_grid_shape.tolist()

        self.num_control_points = np.prod(self.cp_grid_shape)
        self.kernel = self.bspline_kernel_3d().expand(3, *((np.ones(3 + 1, dtype=int) * -1).tolist()))
        self.kernel_size = np.asarray(self.kernel.size())[2:]
        self.padding = ((self.kernel_size - 1) / 2).astype(dtype=int).tolist()

        # Network params
        num_features = torch.prod(((((torch.tensor(input_size) - 4) / 2 - 4) / 2) - 4) / 2)
        self.conv1 = nn.Conv3d(input_channels, 8, kernel_size=5).to(self.device)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=5).to(self.device)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=5).to(self.device)
        self.fc = nn.Linear(32 * num_features, self.num_control_points).to(self.device)

    def gen_3d_mesh_grid(self, d, h, w):
        # move into self to save compute?
        d_s = torch.linspace(-1, 1, d)
        h_s = torch.linspace(-1, 1, h)
        w_s = torch.linspace(-1, 1, w)

        d_s, h_s, w_s = torch.meshgrid([d_s, h_s, w_s])

        mesh_grid = torch.stack([w_s, h_s, d_s])
        return mesh_grid.permute(1, 2, 3, 0).to(self.device)  # d x h x w x 3

    def bspline_kernel_3d(self, order=3):
        kernel_ones = torch.ones(1, 1, *self.control_point_spacing)
        kernel = kernel_ones

        for i in range(1, order + 1):
            kernel = F.conv3d(kernel, kernel_ones, padding=self.control_point_spacing.tolist()) / self.area

        return kernel.to(dtype=self.dtype, device=self.device)

    def compute_displacement(self, params):
        # compute dense displacement
        displacement = F.conv_transpose3d(params, self.kernel,
                                          padding=self.padding, stride=self.stride, groups=3)

        # crop displacement
        displacement = displacement[:, :,
                       self.control_point_spacing[0] + self.crop_start[0]:-self.control_point_spacing[0] -
                                                                          self.crop_end[0],
                       self.control_point_spacing[1] + self.crop_start[1]:-self.control_point_spacing[1] -
                                                                          self.crop_end[1],
                       self.control_point_spacing[2] + self.crop_start[2]:-self.control_point_spacing[2] -
                                                                          self.crop_end[2]]

        return displacement.permute(0, 2, 3, 4, 1)

    def get_theta(self, i):
        return self.control_points[i]

    def forward(self, img):
        
        x = F.interpolate(img, size = self.input_size, mode = 'trilinear', align_corners = False)
        b, c, d, h, w = x.shape
        xs = F.avg_pool3d(F.relu(self.conv1(x)), 2)
        xs = F.avg_pool3d(F.relu(self.conv2(xs)), 2)
        xs = F.avg_pool3d(F.relu(self.conv3(xs)), 2)
        xs = xs.view(xs.size(0), -1)
        self.regularisation_loss = 30.0 * torch.mean(torch.abs(xs))
        # cap the displacement field by (-1,1) this still allows for non-diffeomorphic transformations
        xs = torch.tanh(self.fc(xs)) * 0.2
        xs = xs.view(-1, *self.cp_grid_shape)

        self.displacement_field = self.compute_displacement(xs) + self.gen_3d_mesh_grid(d, h, w).unsqueeze(0)
        # extract first channel for warping
        # img = x.narrow(dim=1, start=0, length=1)
        # warped_img = self.warp_image(img).to(self.device)
        
        return img

    def warp_image(self, img):
        
        tem_displacement_field = self.displacement_field.permute(0, 4, 1, 2, 3)
        tem_displacement_field = F.interpolate(tem_displacement_field, size = img.shape[-3:], mode = 'trilinear', align_corners = False)
        tem_displacement_field = tem_displacement_field.permute(0, 2, 3, 4, 1)
        img = F.grid_sample(img, tem_displacement_field)
        
        return img
        
    def unwarp_image(self, img):
        
        tem_displacement_field = self.displacement_field.permute(0, 4, 1, 2, 3)
        tem_displacement_field = F.interpolate(tem_displacement_field, size = img.shape[-3:], mode = 'trilinear', align_corners = False)
        tem_displacement_field = tem_displacement_field.permute(0, 2, 3, 4, 1)
        
        original_shape = img.shape
        batch_size, in_channels, _, _, _ = original_shape
        
        iz = (tem_displacement_field[..., 0].reshape(-1) + 1.0) * original_shape[-3] / 2.0
        iy = (tem_displacement_field[..., 1].reshape(-1) + 1.0) * original_shape[-2] / 2.0
        ix = (tem_displacement_field[..., 2].reshape(-1) + 1.0) * original_shape[-1] / 2.0
        
        x0 = torch.clamp(ix.floor().int(), 0, original_shape[-3] - 1).reshape(-1)
        x1 = torch.clamp(x0 + 1, 0, original_shape[-3] - 1).reshape(-1)
        y0 = torch.clamp(iy.floor().int(), 0, original_shape[-2] - 1).reshape(-1)
        y1 = torch.clamp(y0 + 1, 0, original_shape[-2] - 1).reshape(-1)
        z0 = torch.clamp(iz.floor().int(), 0, original_shape[-1] - 1).reshape(-1)
        z1 = torch.clamp(z0 + 1, 0, original_shape[-1] - 1).reshape(-1)
        
        
        dim0 = original_shape[-3] * original_shape[-2] * original_shape[-1]
        dim1 = original_shape[-2] * original_shape[-1]
        dim2 = original_shape[-1]
        dimb = batch_size * dim0
        
        base = (torch.arange(batch_size) * dim0).unsqueeze(1).repeat(1, dim0).reshape(-1).int().to(self.device)
        
        idx_a = (base + x0 * dim1 + y0 * dim2 + z0).long().unsqueeze(1)
        idx_b = (base + x0 * dim1 + y0 * dim2 + z1).long().unsqueeze(1)
        idx_c = (base + x0 * dim1 + y1 * dim2 + z0).long().unsqueeze(1)
        idx_d = (base + x0 * dim1 + y1 * dim2 + z1).long().unsqueeze(1)
        idx_e = (base + x1 * dim1 + y0 * dim2 + z0).long().unsqueeze(1)
        idx_f = (base + x1 * dim1 + y0 * dim2 + z1).long().unsqueeze(1)
        idx_g = (base + x1 * dim1 + y1 * dim2 + z0).long().unsqueeze(1)
        idx_h = (base + x1 * dim1 + y1 * dim2 + z1).long().unsqueeze(1)
        
        im_flat = img.reshape(batch_size, in_channels, dim0).transpose(1, 2).reshape(dimb, in_channels)
        
        Ia = torch.zeros(dimb, in_channels).to(self.device).scatter_add_(0, idx_a.repeat(1, in_channels), im_flat)
        Ib = torch.zeros(dimb, in_channels).to(self.device).scatter_add_(0, idx_b.repeat(1, in_channels), im_flat)
        Ic = torch.zeros(dimb, in_channels).to(self.device).scatter_add_(0, idx_c.repeat(1, in_channels), im_flat)
        Id = torch.zeros(dimb, in_channels).to(self.device).scatter_add_(0, idx_d.repeat(1, in_channels), im_flat)
        Ie = torch.zeros(dimb, in_channels).to(self.device).scatter_add_(0, idx_e.repeat(1, in_channels), im_flat)
        If = torch.zeros(dimb, in_channels).to(self.device).scatter_add_(0, idx_f.repeat(1, in_channels), im_flat)
        Ig = torch.zeros(dimb, in_channels).to(self.device).scatter_add_(0, idx_g.repeat(1, in_channels), im_flat)
        Ih = torch.zeros(dimb, in_channels).to(self.device).scatter_add_(0, idx_h.repeat(1, in_channels), im_flat)
        
        dx0 = torch.clamp(torch.abs(ix - x0.float()), 0, 1)
        dx1 = torch.clamp(torch.abs(x1.float() - ix), 0, 1)
        dy0 = torch.clamp(torch.abs(iy - y0.float()), 0, 1)
        dy1 = torch.clamp(torch.abs(y1.float() - iy), 0, 1)
        dz0 = torch.clamp(torch.abs(iz - z0.float()), 0, 1)
        dz1 = torch.clamp(torch.abs(z1.float() - iz), 0, 1)
        
        wa = torch.zeros(dimb, 1).to(self.device).scatter_add_(0, idx_a, (dx1 * dy1 * dz1).unsqueeze(1))
        wb = torch.zeros(dimb, 1).to(self.device).scatter_add_(0, idx_b, (dx1 * dy1 * dz0).unsqueeze(1))
        wc = torch.zeros(dimb, 1).to(self.device).scatter_add_(0, idx_c, (dx1 * dy0 * dz1).unsqueeze(1))
        wd = torch.zeros(dimb, 1).to(self.device).scatter_add_(0, idx_d, (dx1 * dy0 * dz0).unsqueeze(1))
        we = torch.zeros(dimb, 1).to(self.device).scatter_add_(0, idx_e, (dx0 * dy1 * dz1).unsqueeze(1))
        wf = torch.zeros(dimb, 1).to(self.device).scatter_add_(0, idx_f, (dx0 * dy1 * dz0).unsqueeze(1))
        wg = torch.zeros(dimb, 1).to(self.device).scatter_add_(0, idx_g, (dx0 * dy0 * dz1).unsqueeze(1))
        wh = torch.zeros(dimb, 1).to(self.device).scatter_add_(0, idx_h, (dx0 * dy0 * dz0).unsqueeze(1))
        
        value_all = wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih
        raw_weight_all = wa + wb + wc + wd + we + wf + wg + wh
        weight_all = torch.clamp(raw_weight_all, 1e-5, 1e+10)
        flag = (weight_all <= 1e-5 * torch.ones(weight_all.shape).to(self.device)).float()
        
        output = (value_all / weight_all) + im_flat * flag
        output = output.clamp(im_flat.min().item(), im_flat.max().item())

        output = output.reshape(batch_size, dim0, in_channels).transpose(1, 2).reshape(original_shape)
        
        return output

