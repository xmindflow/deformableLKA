###############################
#
# STViT
#
###############################
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
from networks.STViTLayers import Block, SemanticAttentionBlock, PosCNN, PatchEmbed, multi_scale_semantic_token1, RestoreBlock
#from STViTLayers import Block, SemanticAttentionBlock, PosCNN, PatchEmbed, multi_scale_semantic_token1, RestoreBlock
import math
#from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from einops import rearrange
from functools import partial
from fvcore.nn import FlopCountAnalysis

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

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

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
            return x, H, W, x_down, Wh, Ww
        else:
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
                 downsample=None, relative_pos=False, use_conv_pos=False, pad_mask=True):
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

    def forward(self, x, H, W):
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
        return x, H, W, x_down, Wh, Ww

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


class SwinTransformer(nn.Module):
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
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24],
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
                 out_indices=(0, 1, 2, 3),
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
            if i_layer in [0, 1, 3]:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    # use_layer_scale=False
                    )

            elif i_layer ==2:
                layer = Deit(
                    embed_dim=int(embed_dim * 2 ** i_layer),
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
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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
            logger = get_root_logger()
            #load_checkpoint(self, pretrained, strict=False, logger=logger)
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

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
    



##################################
#
# LKA Modules
#
##################################

from timm.models.layers import DropPath

class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class LKAMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) #build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim) #build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LKAMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        y = x.permute(0,2,3,1) # b h w c, because norm requires this
        y = self.norm1(y)
        y = y.permute(0, 3, 1, 2) # b c h w, because attn requieres this
        y = self.attn(y)
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        #x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
        #                       * self.attn(self.norm1(x)))

        y = x.permute(0,2,3,1) # b h w c, because norm requires this
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2) # b c h w, because attn requieres this
        y = self.mlp(y)
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        #x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
        #                       * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        #print("LKA return shape: {}".format(x.shape))
        return x


##################################
#
# End of LKA Modules
#
##################################




## Decoder
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


class MyDecoderLayer(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        #print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here, skip connections
            #self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here, skip connections
            #self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_lka_1 = LKABlock(dim=out_dim) # TODO
        self.layer_lka_2 = LKABlock(dim=out_dim) # TODO

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b, -1, c) # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1) # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128
            
            cat_linear_x = x1_expand + x2 # Simply add them in the first step. TODO: Add more complex skip connection here

            tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_lka_1(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2)) # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2) # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out
    
##############################################
#
# STVit Encoder LKA Decoder segmentation model
#
##############################################

class STVitLKA(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        dims, key_dim, value_dim, layers = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2]]
        self.backbone = SwinTransformer()

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        #in_out_chan = [
        #    [64, 128, 128, 128, 160],
        #    [320, 320, 320, 320, 256],
        #    [512, 512, 512, 512, 512],
        #]  # [dim, out_dim, key_dim, value_dim, x2_dim]

        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]

        self.decoder_3 = MyDecoderLayer(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )

        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.backbone(x)

        b, c, _, _ = output_enc[3].shape

        # ---------------Decoder-------------------------
        #print("Input to decoder 3")
        tmp_3 = self.decoder_3(output_enc[3].permute(0, 2, 3, 1).view(b, -1, c))
        #print("Input to decoder 2")
        tmp_2 = self.decoder_2(tmp_3, output_enc[2].permute(0, 2, 3, 1))
        #print("Input to decoder 1")
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        #print("Input to decoder 0")
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))

        return tmp_0


if __name__ == "__main__":
    

    input = torch.rand((1,3,224,224)).cuda(0)

    net = STVitLKA().cuda(0)
    
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    flops = FlopCountAnalysis(net, input)
    model_flops = flops.total()
    print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
    print(f"MAdds: {round(model_flops * 1e-9, 2)} G")

    output = net(input)
    print("Out shape: " + str(output.shape))