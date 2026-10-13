# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from STViTLayers import Block, SemanticAttentionBlock, PosCNN, PatchEmbed, multi_scale_semantic_token1, RestoreBlock, PatchExpand, FinalPatchExpand_X4
import math
#from ..builder import BACKBONES
#from mmcv_custom import load_checkpoint
#from mmdet.utils import get_root_logger


###############################
# Start of STVit Layers
###############################

""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
import numpy as np
import pdb
import pickle

_logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., relative_pos=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.relative_pos = relative_pos
        if self.relative_pos:
            self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=.02)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, y, mask=None):
        B, N1, C = x.shape
        B, N2, C = y.shape
        q = self.q(x).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_pos:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask != None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, window_size=3, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5, relative_pos=False, local=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, relative_pos=relative_pos)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.local = local
        self.window_size = window_size
        self.H = None
        self.W = None

    def forward(self, x):
        shortcut = x  # B, L, C
        x = self.norm1(x)
        if self.local:
            B, L, C = x.shape
            assert L == self.H * self.W, "input feature has wrong size"
            x = x.view(B, self.H, self.W, C)

            x = x.view(B, self.H // self.window_size, self.window_size, self.W // self.window_size, self.window_size, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size*self.window_size, C)
            attn = self.attn(x, x)
            attn = attn.view(B, self.H // self.window_size, self.W // self.window_size, self.window_size, self.window_size, C)
            attn = attn.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, C)
        else:
            attn = self.attn(x, x)
        x = shortcut + self.drop_path(self.layer_scale_1 * attn)
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class SemanticAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, multi_scale, window_size=7, window_sample_size=3, k_window_size=14, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5, 
                 use_conv_pos=False, pad_mask=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.multi_scale = multi_scale(window_sample_size)
        self.attn = Attention(dim, num_heads=num_heads, window_size=None, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.use_conv_pos = use_conv_pos
        if self.use_conv_pos:
            self.conv_pos = PosCNN(dim, dim)
        self.window_size = window_size
        self.window_sample_size = window_sample_size
        self.k_window_size = k_window_size
        self.pad_mask = pad_mask
        self.H = None
        self.W = None

    def forward(self, x, y=None):
        """
        x: image token as key & vale
        y: semantic token as query. If y is None, semantic token is generated from x.
        """
        B, L, C = x.shape
        assert L == self.H * self.W, "input feature has wrong size"
        x = x.view(B, self.H, self.W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - self.W % self.window_size) % self.window_size
        pad_b = (self.window_size - self.H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        num_samples = (Hp // self.window_size * self.window_sample_size, Wp // self.window_size * self.window_sample_size)
        if y == None:
            xx = x.reshape(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
            windows = xx.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, self.window_size, self.window_size, C).permute(0, 3, 1, 2)
            shortcut = self.multi_scale(windows)  # B*nW, W*W, C
            if self.use_conv_pos:
                shortcut = self.conv_pos(shortcut)
            pool_x = self.norm1(shortcut.reshape(B, -1, C)).reshape(-1, self.multi_scale.num_samples, C)
        else:
            B, L_, C = y.shape
            assert L_ == num_samples[0] * num_samples[1], "input feature has wrong size"
            y = y.reshape(B, num_samples[0] // self.window_sample_size, self.window_sample_size, num_samples[1] // self.window_sample_size, self.window_sample_size, C)
            y = y.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_sample_size*self.window_sample_size, C)
            shortcut = y
            if self.use_conv_pos:
                shortcut = self.conv_pos(shortcut)
            pool_x = self.norm1(shortcut.reshape(B, -1, C)).reshape(-1, self.multi_scale.num_samples, C)
        
        # produce K, V
        left = math.floor((self.k_window_size - self.window_size) / 2)
        right = math.ceil((self.k_window_size - self.window_size) / 2)
        xx = F.pad(x, (0, 0, left, right, left, right))
        if self.pad_mask:
            pad_mask = torch.zeros(x.size()[:-1], device=x.device)
            pad_mask[:, -pad_b:, -pad_r:] = -1000  # float('-inf')
            pad_mask = F.pad(pad_mask, (left, right, left, right), value=-1000)
            pad_mask = F.unfold(pad_mask.unsqueeze(1), kernel_size=self.k_window_size, stride=self.window_size).view(B, self.k_window_size, self.k_window_size, -1).permute(0, 3, 1, 2)
            pad_mask = pad_mask.reshape(-1, 1, self.k_window_size*self.k_window_size)
            pad_mask = pad_mask.expand(-1, self.multi_scale.num_samples, -1)  # B, num_samples, self.k_window_size**2
            pad_mask = pad_mask.unsqueeze(1)
        else:
            pad_mask = None
        k_windows = F.unfold(xx.permute(0, 3, 1, 2), kernel_size=self.k_window_size, stride=self.window_size).view(B, C, self.k_window_size, self.k_window_size, -1).permute(0, 4, 2, 3, 1)
        k_windows = k_windows.reshape(-1, self.k_window_size*self.k_window_size, C)
        # k_windows = torch.cat([shortcut, k_windows], dim=1)
        k_windows = self.norm1(k_windows.reshape(B, -1, C)).reshape(-1, self.k_window_size*self.k_window_size, C)

        x = shortcut + self.drop_path(self.layer_scale_1 * self.attn(pool_x, k_windows, mask=pad_mask))
        x = x.view(B, Hp // self.window_size, Wp // self.window_size, self.window_sample_size, self.window_sample_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, C)
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x, num_samples[0], num_samples[1]


class RestoreBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, window_sample_size=3, k_window_size=27, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5, pad_mask=True
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, window_size=None, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.window_size = window_size
        self.window_sample_size = window_sample_size
        self.k_window_size = k_window_size
        self.pad_mask = pad_mask
        self.H = None
        self.W = None

    def forward(self, x, y):
        """
        x: image token as query
        y: semantic token as key & value
        """
        B, L, C = x.shape
        assert L == self.H * self.W, "input feature has wrong size"
        x = x.view(B, self.H, self.W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - self.W % self.window_size) % self.window_size
        pad_b = (self.window_size - self.H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        x = x.reshape(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size*self.window_size, C)
        shortcut = x
        x = self.norm1(shortcut.reshape(B, -1, C)).reshape(-1, self.window_size*self.window_size, C)

        num_samples = (Hp // self.window_size * self.window_sample_size, Wp // self.window_size * self.window_sample_size)

        B, L_, C = y.shape
        assert L_ == num_samples[0] * num_samples[1], "input feature has wrong size"
        y = y.reshape(B, num_samples[0], num_samples[1], C)
        # y = y.reshape(B, num_samples[0] // self.window_sample_size, self.window_sample_size, num_samples[1] // self.window_sample_size, self.window_sample_size, C)
        # y = y.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_sample_size*self.window_sample_size, C)
        # shortcut = y
        # pool_x = self.norm1(shortcut.reshape(B, -1, C)).reshape(-1, self.multi_scale.num_samples, C)
        
        # produce K, V
        left = math.floor((self.k_window_size - self.window_sample_size) / 2)
        right = math.ceil((self.k_window_size - self.window_sample_size) / 2)

        if self.pad_mask:
            pad_mask = torch.zeros(y.size()[:-1], device=y.device, requires_grad=False)
            pad_mask[:, -pad_b:, -pad_r:] = -1000 # float('-inf')
            pad_mask = F.pad(pad_mask, (left, right, left, right), value=-1000)
            pad_mask = F.unfold(pad_mask.unsqueeze(1), kernel_size=self.k_window_size, stride=self.window_sample_size).view(B, self.k_window_size, self.k_window_size, -1).permute(0, 3, 1, 2)
            pad_mask = pad_mask.reshape(-1, 1, self.k_window_size*self.k_window_size)
            pad_mask = pad_mask.expand(-1, self.window_size**2, -1)  # B, num_samples, self.k_window_size**2
            pad_mask = pad_mask.unsqueeze(1)
        else:
            pad_mask = None
        y = F.pad(y, (0, 0, left, right, left, right))
        k_windows = F.unfold(y.permute(0, 3, 1, 2), kernel_size=self.k_window_size, stride=self.window_sample_size).view(B, C, self.k_window_size, self.k_window_size, -1).permute(0, 4, 2, 3, 1)
        k_windows = self.norm1(k_windows.reshape(B, -1, C)).reshape(-1, self.k_window_size*self.k_window_size, C)

        x = shortcut + self.drop_path(self.layer_scale_1 * self.attn(x, k_windows, mask=pad_mask))
        x = x.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # .view(B, -1, C)
        if pad_r > 0 or pad_b > 0:
            x = x.view(B, Hp, Wp, C)[:, :self.H, :self.W, :].contiguous()
        x = x.reshape(B, self.H*self.W, C)
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, H)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(out_channels)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding

#     Args:
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         patch_size = to_2tuple(patch_size)
#         self.patch_size = patch_size

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         """Forward function."""
#         # padding
#         _, _, H, W = x.size()
#         if W % self.patch_size[1] != 0:
#             x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
#         if H % self.patch_size[0] != 0:
#             x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

#         x = self.proj(x)  # B C Wh Ww
#         if self.norm is not None:
#             Wh, Ww = x.size(2), x.size(3)
#             x = x.flatten(2).transpose(1, 2)
#             x = self.norm(x)
#             x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

#         return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # self.img_size = img_size
        self.patch_size = patch_size
        # self.patches_resolution = patches_resolution
        # self.num_patches = patches_resolution[0] * patches_resolution[1]

        # self.in_chans = in_chans
        # self.embed_dim = embed_dim

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim // 2, kernel_size=3, stride=2, pad=1),
            torch.nn.Hardswish(),
            Conv2d_BN(embed_dim // 2, embed_dim, kernel_size=3, stride=2, pad=1),
            torch.nn.Hardswish(),
        )
        # if norm_layer is not None:
        #     self.norm = norm_layer(embed_dim)
        # else:
        #     self.norm = None

    def forward(self, x):
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        # x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        # if self.norm is not None:
        #     x = self.norm(x)
        return x


class multi_scale_semantic_token1(nn.Module):
    def __init__(self, sample_window_size):
        super().__init__()
        self.sample_window_size = sample_window_size
        self.num_samples = sample_window_size * sample_window_size

    def forward(self, x):
        B, C, _, _ = x.size()
        pool_x = F.adaptive_max_pool2d(x, (self.sample_window_size, self.sample_window_size)).view(B, C, self.num_samples).transpose(2, 1)
        return pool_x
    
from einops import rearrange    
    
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x

###############################
# End of STViT Layers
###############################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        #print("Created norm1 layer with dim: "+ str(dim))
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        #print("Current x shape before norm1:"+ str(x.shape))
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        upsample (nn.Module | None, optional): Upsample layer at the start of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 upsample=None,
                 input_res=(224,224),
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # patch expanding layer
        if upsample is not None:
            self.upsample = upsample(input_resolution= input_res, dim=dim, norm_layer=norm_layer) # TODO: set input resolution x: B, H*W, C
        else:
            self.upsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        #print("Forward of BasicBlock")
        #print("Input X shape: " + str(x.shape))

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            #print("Downsampling. X shape: "+ str(x.shape))
            #print("X down shape: " + str(x_down.shape))
            return x, H, W, x_down, Wh, Ww
        

        if self.upsample is not None:
            x_up = self.upsample(x)
            Hd = H * 2
            Wd = W * 2
            #print("Upsampling. New X shape: " + str(x_up.shape))
            return x, H, W, x_up, Hd, Wd

        else:
            #print("No Downsampling or upsampling. X shape: "+ str(x.shape))
            return x, H, W, x, H, W


class Deit(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, window_size, window_sample_size, k_window_size_1, k_window_size_2, 
                 restore_k_window_size, multi_scale, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='', semantic_key_concat=False,
                 downsample=None, upsample=None, input_res=(224,224), relative_pos=False, use_conv_pos=False, pad_mask=True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            upsample (nn.Module | None, optional): Upsample layer at the start of the layer. Default: None
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.multi_scale = multi_scale
        self.semantic_key_concat = semantic_key_concat
        self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i in [0, 6, 12]:
                self.blocks.append(SwinTransformerBlock(dim=embed_dim,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i],
                    norm_layer=norm_layer)
                    )
            elif i in [1, 7, 13]:
                self.blocks.append(SemanticAttentionBlock(
                    dim=embed_dim, window_size=window_size, window_sample_size=window_sample_size, k_window_size=k_window_size_1,
                    num_heads=num_heads, multi_scale=self.multi_scale, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[i], 
                    norm_layer=norm_layer, act_layer=act_layer, use_conv_pos=use_conv_pos, pad_mask=pad_mask)
                    )
            elif i in [2, 8, 14]:
                self.blocks.append(SemanticAttentionBlock(
                    dim=embed_dim, window_size=window_size, window_sample_size=window_sample_size, k_window_size=k_window_size_2,
                    num_heads=num_heads, multi_scale=self.multi_scale, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[i], 
                    norm_layer=norm_layer, act_layer=act_layer, pad_mask=pad_mask)
                    )
            elif i in [5, 11, 17]:
                self.blocks.append(RestoreBlock(
                    dim=embed_dim, window_size=window_size, window_sample_size=window_sample_size, k_window_size=restore_k_window_size, 
                    num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                    drop=drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer, act_layer=act_layer, pad_mask=pad_mask)
                    )
            else:
                self.blocks.append(Block(
                    dim=embed_dim, window_size=window_sample_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                    drop=drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer, act_layer=act_layer, relative_pos=relative_pos, local=bool(i%2))
                    )
        if downsample is not None:
            self.downsample = downsample(dim=self.embed_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # patch expanding layer
        if upsample is not None:
            self.upsample = upsample(input_resolution=input_res, dim=embed_dim, norm_layer=norm_layer) # TODO: set input resolution x: B, H*W, C
        else:
            self.upsample = None

    def forward(self, x, H, W):

        #print("Forward of DeiT")
        #print("Input x shape: "+ str(x.shape))

        for i, blk in enumerate(self.blocks):
            if i == 0:
                blk.H, blk.W = H, W
                x = blk(x, mask_matrix=None)
            elif i == 1:
                blk.H, blk.W = H, W
                semantic_token, s_H, s_W = blk(x)
            elif i == 2:
                blk.H, blk.W = H, W
                semantic_token, _, _ = blk(x, semantic_token)
            elif i > 2 and i < 5:
                blk.H, blk.W = s_H, s_W
                semantic_token = blk(semantic_token)
            elif i == 5:
                blk.H, blk.W = H, W
                x = blk(x, semantic_token)
            elif i == 6:
                blk.H, blk.W = H, W
                x = blk(x, mask_matrix=None)
            elif i == 7:
                blk.H, blk.W = H, W
                semantic_token, _, _ = blk(x)
            elif i == 8:
                blk.H, blk.W = H, W
                semantic_token, _, _ = blk(x, semantic_token)
            elif i > 8 and i < 11:
                blk.H, blk.W = s_H, s_W
                semantic_token = blk(semantic_token)
            elif i == 11:
                blk.H, blk.W = H, W
                x = blk(x, semantic_token)
            elif i == 12:
                blk.H, blk.W = H, W
                x = blk(x, mask_matrix=None)
            elif i == 13:
                blk.H, blk.W = H, W
                semantic_token, _, _ = blk(x)
            elif i == 14:
                blk.H, blk.W = H, W
                semantic_token, _, _ = blk(x, semantic_token)
            elif i > 14 and i < 17:
                blk.H, blk.W = s_H, s_W
                semantic_token = blk(semantic_token)
            else:
                blk.H, blk.W = H, W
                x = blk(x, semantic_token)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            #print("Downsampling. Output x shape: " + str(x.shape))
            #print("Xdown shape: "+ str(x_down.shape))
            #return x, H, W, x_down, Wh, Ww
            return x, H, W, x_down, Wh, Ww
        
        elif self.upsample is not None:
            x_up = self.upsample(x)
            #print("Upsampling. X shape: "+ str(x_up.shape))
            Hd = H * 2
            Wd = W * 2
            return x, H, W, x_up, Wd, Hd

        else:
            #print("Output X shape: " + str(x.shape))
            return x, H, W, x, Wh, Ww

    def flops(self):
        flops = 0
        return flops


class Deit2(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, input_resolution, num_samples, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='', semantic_key_concat=False,
                 downsample=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_samples = num_samples
        self.semantic_key_concat = semantic_key_concat
        self.input_resolution = input_resolution
        self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.semantic_token = nn.Parameter(torch.zeros(1, self.num_samples, embed_dim))
        # trunc_normal_(self.semantic_token, std=.02)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, window_size=(int(math.sqrt(self.num_samples)), int(math.sqrt(self.num_samples))), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=self.embed_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, s):
        s, _ = self.blocks[0](s, s)
        s, _ = self.blocks[1](s, s)
        if self.downsample is not None:
            s = self.downsample(s)
        return s

    def flops(self):
        flops = 0
        return flops


class SemanticSTViT(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, 
                 pretrain_img_size=224, 
                 patch_size=4, 
                 in_chans=3,
                 embed_dim=96, 
                 num_classes=9,
                 depths= [2, 2, 6, 6, 2, 2, 2], #[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24, 12, 6, 3], #[3, 6, 12, 24],
                 window_size=7, 
                 window_sample_size=3, 
                 k_window_size_1=14,
                 k_window_size_2=21, 
                 restore_k_window_size=27,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 ape=False, 
                 patch_norm=True,
                 use_checkpoint=False, 
                 out_indices=(0,1,2,3,4,5,6), # (0, 1, 2, 3),
                 frozen_stages=-1,
                 multi_scale='multi_scale_semantic_token1', 
                 relative_pos=False, use_conv_pos=False,
                 # use_layer_scale=False, 
                 pad_mask=True):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.multi_scale = eval(multi_scale)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer in [0, 1]: # Swin Encoder layers
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), # 96 192 
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, # No Downsampling in Basiclayer 3!
                    use_checkpoint=use_checkpoint,
                    # use_layer_scale=False
                    )

            elif i_layer in [2]: # Deit Encoder layer with semantic tokens
                layer = Deit(
                    embed_dim=int(embed_dim * 2 ** i_layer), # 384
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    window_sample_size=window_sample_size,
                    k_window_size_1=k_window_size_1,
                    k_window_size_2=k_window_size_2,
                    restore_k_window_size=restore_k_window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    multi_scale=self.multi_scale,
                    relative_pos=relative_pos,
                    use_conv_pos=use_conv_pos,
                    pad_mask=pad_mask
                )

            elif i_layer in [3]:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), #768 
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    #downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, # No Downsampling in Basiclayer 3!
                    upsample=PatchExpand,
                    input_res=(7,7),
                    use_checkpoint=use_checkpoint,
                    # use_layer_scale=False
                    )
                
            elif i_layer in [4]: # Deit Decoder layer with semantic tokens
                layer = Deit(
                    embed_dim=int(embed_dim * 2 ** (i_layer-2)), # 384
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    window_sample_size=window_sample_size,
                    k_window_size_1=k_window_size_1,
                    k_window_size_2=k_window_size_2,
                    restore_k_window_size=restore_k_window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    #downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, # TODO: No Downsampling here! Instead upsampling earlier!
                    upsample=PatchExpand,
                    input_res=(14,14),
                    multi_scale=self.multi_scale,
                    relative_pos=relative_pos,
                    use_conv_pos=use_conv_pos,
                    pad_mask=pad_mask
                )    
                
            elif i_layer in [5]: # Swin Decoder layer
                layer = BasicLayer(dim=int(embed_dim * 2 ** (i_layer - 4)), # 192
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    #downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    upsample=PatchExpand,
                    input_res=(28,28),
                    use_checkpoint=use_checkpoint,
                    # use_layer_scale=False
                    )
            
            elif i_layer in [6]: # Swin Decoder layer
                #print("Dim for Basiclayer 7: "+ str(int(embed_dim * 2 ** (i_layer - 6))))
                layer = BasicLayer(dim=int(embed_dim * 2 ** (i_layer - 6)), #96
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    #downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    upsample=FinalPatchExpand_X4,
                    input_res=(56,56),
                    use_checkpoint=use_checkpoint,
                    # use_layer_scale=False
                    )
                 
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        num_features = [96, 192, 384, 768, 384, 192, 96]
        #print("Num features: "+ str(num_features))
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            #print("Create norm layer: " + str(layer))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.last_layer = nn.Conv2d(num_features[-1], num_classes, 1)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        """Forward function."""

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)


        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                #print("XOut shape: " + str(x_out.shape))
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
            #print("Out shape: " + str(out.size()))

        # reshape x so fit b c h w
        x_final = x.view(-1, H*4, W*4, self.num_features[-1]).permute(0, 3, 1, 2).contiguous()
        x_final = self.last_layer(x_final)
        outs.append(x_final)
        #print("Final output shape: " + str(x_final.shape))

        #return tuple(outs)
        return x_final

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


if __name__ == '__main__':
    input = torch.rand((1,3,224,224)).cuda(0)
    print("Created Sample input Tensor")
    net = SemanticSTViT().cuda(0)
    print("Created network")
    
    output = net(input)
    