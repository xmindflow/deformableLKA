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