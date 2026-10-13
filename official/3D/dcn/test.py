#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
from modules.deform_conv import DeformConv_d, _DeformConv, DeformConvPack_d
#from dcn.modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
#from dcn.modules.deform_conv import DeformConv_d, _DeformConv, DeformConvPack_d

deformable_groups = 32
B, inC, inT, inH, inW = 2, 32, 32, 32, 32
outC = 32
kT, kH, kW = 5, 5, 5
sT, sH, sW = 1, 1, 1
pT, pH, pW = 2, 2, 2
dT, dH, dW = 1, 1, 1


class deform_LKA3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConvPack(dim, dim, 5,stride=1, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class deform_LKA_Attention3d(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deform_LKA3d(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x



def example_dconv():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    dcn = DeformConvPack(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_offset():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    offset = torch.randn(B, kT*kH*kW*3, inT, inH, inW).cuda()
    dcn = DeformConv(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW], dilation=[dT, dH, dW]).cuda()
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_d():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    dcn = DeformConvPack_d(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW],dimension='TW').cuda()
    #  dimension = 'T' or 'H' or 'W' or any combination of these three letters
    #  'T' represents the deformation in temporal dimension
    #  'H' represents the deformation in height dimension
    #  'W' represents the deformation in weigh dimension
    print('input.shape: ', input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_offset_d():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    #  offset
    dimension = 'HW' # choose any dimension you want to deform
    offset = torch.randn(B, kT*kH*kW*len(dimension), inT, inH, inW).cuda()
    dcn = DeformConv_d(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW],dimension=dimension).cuda()
    #  dimension = 'T' or 'H' or 'W' or any combination of these three letters
    #  'T' represents the deformation in temporal dimension
    #  'H' represents the deformation in height dimension
    #  'W' represents the deformation in weigh dimension
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

if __name__ == '__main__':
    print('==================Deformable 3D convolution==================', '\n')
    # D3D deform in three dimensions
    print('=============D3D deform in three dimensions===========')
    #example_dconv() # DCN using its own offsets
    #example_dconv_offset() # DCN using extra offsets
    print('\n')

    # D3D available for deformable dimension
    print('============option for deformable dimension===========')
    #example_dconv_d()  # DCN using its own offsets
    #example_dconv_offset_d()  # DCN using extra offsets

    print('============Deform LKA 3D=============================')

    input = torch.rand(1,32,32,32,32).cuda(0)
    layer = deform_LKA_Attention3d(d_model=32).cuda(0)
    output = layer(input)
    print('output.shape: ', output.shape)
    print("Max memory: {}".format(torch.cuda.max_memory_allocated()/(1024*1024))) # MB


