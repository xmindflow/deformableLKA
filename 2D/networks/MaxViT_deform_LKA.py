import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

#from .networks.segformer import *
from .segformer import *
#from segformer import *


import math
##################################
#
# LKA Modules
#
##################################

from timm.models.layers import DropPath, to_2tuple
#from deformable_LKA.deformable_LKA import deformable_LKA_Attention
from .deformable_LKA import deformable_LKA_Attention

class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
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
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
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


class deformableLKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) #build_norm_layer(norm_cfg, dim)[1]
        self.attn = deformable_LKA_Attention(dim)
        #self.attn = DAttentionBaseline()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim) #build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

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

class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)

        attn = self.attn(norm_1, norm_2)
        # attn = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(attn)

        # residual1 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x1)
        # residual2 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x2)
        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        ## Here channel weighting and Eigenvalues
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        #print("Dual transformer return shape: {}".format(mx.shape))
        return mx


# Encoder
class MiT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


# Decoder
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
## CBAM + SRM
## By: Sina 
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
class EfCBAM(nn.Module):
    def __init__(self, gate_channels, no_spatial=False):
        super(EfCBAM, self).__init__()
        #self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.ChannelGate = ECALayer(gate_channels, gamma=2, b = 1)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)


class Gated_Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Gated_Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int) #Gate - Signal (Encoder)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int) #X - Signal (Decoder)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        ) # The Second part of attn

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


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
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            #self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            #)
            #self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            #self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            #)
            #self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        #self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        #self.layer_lka_1 = deformableLKABlock(dim=out_dim) # TODO
        self.layer_lka_1 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)
        self.layer_lka_2 =nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)
        #self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        #self.layer_lka_2 = deformableLKABlock(dim=out_dim) # TODO

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

            #b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2) # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1) # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            #cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2)) # 1 784 320, 1 3136 128
            cat_linear_x = x1_expand + x2 # Simply add them in the first step. TODO: Add more complex skip connection here
            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1) // w2,
                                             cat_linear_x.size(1) // h2) # B x C x H x W
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w) # 1 784 320, 1 3136 128
            #tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            tran_layer_2 = self.layer_lka_2(cat_linear_x)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w) # 1 784 320, 1 3136 128
            #tran_layer_2 = self.layer_lka_2(tran_layer_1, h, w) #self.layer_lka_1(tran_layer_1, h, w) # Here has to be a 2! LEON CHANGE THIS!!!!
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2)) # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2) # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out

##########################################
#
# Decoder2 : DLKA --> CBAM
# By: Sina 
##########################################

class MyDecoderLayer2(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        #print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            #self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            #)
            #self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            #self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            #)
            #self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        #self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        #self.layer_lka_1 = deformableLKABlock(dim=out_dim) # TODO


        #self.layer_lka_1 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)
        #self.layer_lka_2 =nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)

        self.layer_lka_1 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        self.layer_lka_2 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)

        #self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        #self.layer_lka_2 = deformableLKABlock(dim=out_dim) # TODO

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

            #b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2) # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1) # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            #cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2)) # 1 784 320, 1 3136 128
            cat_linear_x = x1_expand + x2 # Simply add them in the first step. TODO: Add more complex skip connection here
            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1) // w2,
                                             cat_linear_x.size(1) // h2)
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w) # 1 784 320, 1 3136 128
            #tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            #print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)
            #print(tran_layer_2.shape)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w) # 1 784 320, 1 3136 128
            #tran_layer_2 = self.layer_lka_2(tran_layer_1, h, w) #self.layer_lka_1(tran_layer_1, h, w) # Here has to be a 2! LEON CHANGE THIS!!!!
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2)) # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2) # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayer3(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            # self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            # self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            # self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            # self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            # self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            # self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        # self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        # self.layer_lka_1 = deformableLKABlock(dim=out_dim) # TODO

        # self.layer_lka_1 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)
        # self.layer_lka_2 =nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)

        self.layer_lka_1 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        self.layer_lka_2 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)

        # self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        # self.layer_lka_2 = deformableLKABlock(dim=out_dim) # TODO

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
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1)  # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            # cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2)) # 1 784 320, 1 3136 128
            # cat_linear_x = x1_expand + x2 # Simply add them in the first step. TODO: Add more complex skip connection here

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            attn_gate = self.ag_attn(x1_expand, x2_new)

            cat_linear_x = x1_expand + attn_gate

            # tran_layer_1 = self.layer_former_1(cat_linear_x, h, w) # 1 784 320, 1 3136 128
            # tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            # print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)
            # print(tran_layer_2.shape)
            # tran_layer_2 = self.layer_former_2(tran_layer_1, h, w) # 1 784 320, 1 3136 128
            # tran_layer_2 = self.layer_lka_2(tran_layer_1, h, w) #self.layer_lka_1(tran_layer_1, h, w) # Here has to be a 2! LEON CHANGE THIS!!!!
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayer4(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        #print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            #self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            #)
            #self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            #self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            #)
            #self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        #self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        #self.layer_lka_1 = deformableLKABlock(dim=out_dim) # TODO


        #self.layer_lka_1 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)
        #self.layer_lka_2 =nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)

        #self.layer_lka_1 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        #self.layer_lka_2 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)

        self.layer_lka_1 = EfCBAM(gate_channels=out_dim)
        self.layer_lka_2 = EfCBAM(gate_channels=out_dim)
        #self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        #self.layer_lka_2 = deformableLKABlock(dim=out_dim) # TODO

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

            #b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2) # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1) # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            #cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2)) # 1 784 320, 1 3136 128
            cat_linear_x = x1_expand + x2 # Simply add them in the first step. TODO: Add more complex skip connection here
            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1) // w2,
                                             cat_linear_x.size(1) // h2)
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w) # 1 784 320, 1 3136 128
            #tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            #print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)
            #print(tran_layer_2.shape)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w) # 1 784 320, 1 3136 128
            #tran_layer_2 = self.layer_lka_2(tran_layer_1, h, w) #self.layer_lka_1(tran_layer_1, h, w) # Here has to be a 2! LEON CHANGE THIS!!!!
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2)) # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2) # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayer5(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            # self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            # self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            # self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            # self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            # self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            # self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        # self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        # self.layer_lka_1 = deformableLKABlock(dim=out_dim) # TODO

        # self.layer_lka_1 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)
        # self.layer_lka_2 =nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)

        # self.layer_lka_1 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        # self.layer_lka_2 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        self.layer_lka_1 = EfCBAM(gate_channels=out_dim)
        self.layer_lka_2 = EfCBAM(gate_channels=out_dim)

        # self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        # self.layer_lka_2 = deformableLKABlock(dim=out_dim) # TODO

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
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1)  # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            # cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2)) # 1 784 320, 1 3136 128
            # cat_linear_x = x1_expand + x2 # Simply add them in the first step. TODO: Add more complex skip connection here

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            attn_gate = self.ag_attn(x1_expand, x2_new)

            cat_linear_x = x1_expand + attn_gate # TODO: nn.parameter (alpha, Beta)

            # tran_layer_1 = self.layer_former_1(cat_linear_x, h, w) # 1 784 320, 1 3136 128
            # tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            # print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)
            # print(tran_layer_2.shape)
            # tran_layer_2 = self.layer_former_2(tran_layer_1, h, w) # 1 784 320, 1 3136 128
            # tran_layer_2 = self.layer_lka_2(tran_layer_1, h, w) #self.layer_lka_1(tran_layer_1, h, w) # Here has to be a 2! LEON CHANGE THIS!!!!
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out

class MyDecoderLayer6(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        #print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            #self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            #)
            #self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            #self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            #)
            #self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        #self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        #self.layer_lka_1 = deformableLKABlock(dim=out_dim) # TODO


        #self.layer_lka_1 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)
        #self.layer_lka_2 =nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)

        #self.layer_lka_1 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        #self.layer_lka_2 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        self.layer_lka_1 = ECALayer(channels=out_dim)
        self.layer_lka_2 = ECALayer(channels=out_dim)
        #self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        #self.layer_lka_2 = deformableLKABlock(dim=out_dim) # TODO

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

            #b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2) # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1) # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            #cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2)) # 1 784 320, 1 3136 128
            cat_linear_x = x1_expand + x2 # Simply add them in the first step. TODO: Add more complex skip connection here
            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1) // w2,
                                             cat_linear_x.size(1) // h2)
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w) # 1 784 320, 1 3136 128
            #tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            #print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)
            #print(tran_layer_2.shape)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w) # 1 784 320, 1 3136 128
            #tran_layer_2 = self.layer_lka_2(tran_layer_1, h, w) #self.layer_lka_1(tran_layer_1, h, w) # Here has to be a 2! LEON CHANGE THIS!!!!
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2)) # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2) # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out

class MyDecoderLayer7(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            # self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            # self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            # self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            # self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            # self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            # self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        # self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        # self.layer_lka_1 = deformableLKABlock(dim=out_dim) # TODO

        # self.layer_lka_1 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)
        # self.layer_lka_2 =nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1,padding=1)

        # self.layer_lka_1 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        # self.layer_lka_2 = CBAM(gate_channels=out_dim, reduction_ratio=reduction_ratio)
        #self.layer_lka_1 = EfCBAM(gate_channels=out_dim)
        #self.layer_lka_2 = EfCBAM(gate_channels=out_dim)

        self.layer_lka_1 = ECALayer(channels=out_dim)
        self.layer_lka_2 = ECALayer(channels=out_dim)
        # self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        # self.layer_lka_2 = deformableLKABlock(dim=out_dim) # TODO

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
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1)  # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            # cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2)) # 1 784 320, 1 3136 128
            # cat_linear_x = x1_expand + x2 # Simply add them in the first step. TODO: Add more complex skip connection here

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            attn_gate = self.ag_attn(x1_expand, x2_new)

            cat_linear_x = x1_expand + attn_gate

            # tran_layer_1 = self.layer_former_1(cat_linear_x, h, w) # 1 784 320, 1 3136 128
            # tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            # print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)
            # print(tran_layer_2.shape)
            # tran_layer_2 = self.layer_former_2(tran_layer_1, h, w) # 1 784 320, 1 3136 128
            # tran_layer_2 = self.layer_lka_2(tran_layer_1, h, w) #self.layer_lka_1(tran_layer_1, h, w) # Here has to be a 2! LEON CHANGE THIS!!!!
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out
##########################################
#
# MaxViT stuff
#
##########################################
#from .merit_lib.networks import MaxViT4Out_Small#, MaxViT4Out_Small3D
#from networks.merit_lib.networks import MaxViT4Out_Small
from .merit_lib.networks import MaxViT4Out_Small
from .merit_lib.decoders import CASCADE_Add, CASCADE_Cat



class MaxViT_deformableLKAFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
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

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        #print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0

##########################################
#
# MaxViT with CBAM as Decoder stuff
# By: Sina
##########################################
class MaxViT_deformableLKAFormer2(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16,8,6,2]
        
        self.decoder_3 = MyDecoderLayer2(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio = reduction_ratio[0])

        self.decoder_2 = MyDecoderLayer2(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio = reduction_ratio[1])
        self.decoder_1 = MyDecoderLayer2(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio = reduction_ratio[2])
        self.decoder_0 = MyDecoderLayer2(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio = reduction_ratio[3])

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        #print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0

class MaxViT_deformableLKAFormer3(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16, 8, 6, 2]

        self.decoder_3 = MyDecoderLayer3(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[0])

        self.decoder_2 = MyDecoderLayer3(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[1])
        self.decoder_1 = MyDecoderLayer3(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayer3(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio=reduction_ratio[3])

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0

class MaxViT_deformableLKAFormer4(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16, 8, 6, 2]

        self.decoder_3 = MyDecoderLayer4(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[0])

        self.decoder_2 = MyDecoderLayer4(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[1])
        self.decoder_1 = MyDecoderLayer4(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayer4(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio=reduction_ratio[3])

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0

class MaxViT_deformableLKAFormer5(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16, 8, 6, 2]

        self.decoder_3 = MyDecoderLayer5(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[0])

        self.decoder_2 = MyDecoderLayer5(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[1])
        self.decoder_1 = MyDecoderLayer5(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayer5(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio=reduction_ratio[3])

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0

class MaxViT_deformableLKAFormer6(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16, 8, 6, 2]

        self.decoder_3 = MyDecoderLayer6(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[0])

        self.decoder_2 = MyDecoderLayer6(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[1])
        self.decoder_1 = MyDecoderLayer6(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayer6(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio=reduction_ratio[3])

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0

class MaxViT_deformableLKAFormer7(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16, 8, 6, 2]

        self.decoder_3 = MyDecoderLayer7(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[0])

        self.decoder_2 = MyDecoderLayer7(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[1])
        self.decoder_1 = MyDecoderLayer7(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayer7(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio=reduction_ratio[3])

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0
if __name__ == "__main__":
    

    input0 = torch.rand((1,3,224,224)).cuda(0)
    input = torch.randn((1,768,7,7)).cuda(0)
    input2 = torch.randn((1,384,14, 14)).cuda(0)
    input3 = torch.randn((1, 192, 28, 28)).cuda(0)
    #b, c, _, _ = input.shape
    dec1 = MyDecoderLayer(input_size=(7,7), in_out_chan= ([768, 768, 768, 768, 768]), head_count=1,
                          token_mlp_mode='mix_skip').cuda(0)
    dec2 = MyDecoderLayer(input_size=(14, 14), in_out_chan=([384, 384, 384, 384, 384]), head_count=1,
                          token_mlp_mode='mix_skip').cuda(0)
    dec3 = MyDecoderLayer(input_size=(28, 28), in_out_chan=([192, 192, 192, 192, 192]), head_count=1,
                          token_mlp_mode='mix_skip').cuda(0)
    #output = dec1(input.permute(0, 2, 3, 1).view(b, -1 , c))
    #output2 = dec2(output, input2.permute(0, 2, 3, 1))
    #output3 = dec3(output2, input3.permute(0, 2, 3, 1))

    net = MaxViT_deformableLKAFormer().cuda(0)

    output0 = net(input0)
    #print("Out shape: " + str(output.shape) + 'Out2 shape:' + str(output2.shape)+"Out shape3: " + str(output3.shape))
    print("Out shape: " + str(output0.shape))
