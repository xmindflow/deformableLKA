import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

#from networks.segformer import *
#from segformer import *
from networks.segformer import *

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
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)

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
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out

#####################################################
#
# BIFORMER STUFF
#
#####################################################
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
#from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

#from ops.bra_legacy import BiLevelRoutingAttention

#from ._common import Attention, AttentionLePE, DWConv
#from _common import Attention, AttentionLePE, DWConv
from networks._common import Attention, AttentionLePE, DWConv

# from positional_encodings import PositionalEncodingPermute2D, Summer
# from siren_pytorch import SirenNet

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        
        return r_weight, topk_index
        

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)
        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel? 
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
    
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v

class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights 
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5


        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        
        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
        
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v 
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad=auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor
        Return:
            NHWC tensor
        """
         # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, # dim=-1
                          pad_l, pad_r, # dim=-2
                          pad_t, pad_b)) # dim=-3
            _, H, W, _ = x.size() # padded size
        else:
            N, H, W, C = x.size()
            assert H%self.n_win == 0 and W%self.n_win == 0 #
        ###################################################


        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x) 

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        
        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out # Added by LEON TODO

def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                       num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                       kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                       topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, mlp_ratio=4, mlp_dwconv=False,
                       side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                        qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                        topk=topk, param_attention=param_attention, param_routing=param_routing,
                                        diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                        auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'), # compatiability
                                      nn.Conv2d(dim, dim, 1), # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim), # pseudo attention
                                      nn.Conv2d(dim, dim, 1), # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                     )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 DWConv(int(mlp_ratio*dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
            

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (N, H, W, C)
            else:
                y = self.norm1(x)
                y = self.attn(y)
                y = self.drop_path(y)
                x = x + y
                #x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x



class BiFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3],
                 in_chans=1,
                 num_classes=1000,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ######## 
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 #-----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1], # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        #if use_checkpoint_stages:
        #    stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(len(embed_dim)-1): #3
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i+1])
            )
            if (pe is not None) and i+1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i+1], name=pe))
            #if use_checkpoint_stages:
            #    downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads= [dim // head_dim for dim in qk_dims]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))] 
        cur = 0
        for i in range(len(embed_dim)): # 4 
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j], 
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            #if i in use_checkpoint_stages:
            #    stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x) # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x

import torch
import torch.nn as nn
#from mmcv_custom import load_checkpoint
#from mmseg.utils import get_root_logger
from timm.models.layers import LayerNorm2d

class BiFormer_mm(BiFormer):
    def __init__(self, pretrained=None, 
                 depth= [4, 4, 18, 4],
                 in_chans=3,
                 num_classes=9,
                 embed_dim= [64, 128, 256, 512],
                 mlp_ratios= [3, 3, 3, 3],
                 n_win= 8,
                 kv_downsample_mode= 'identity',
                 kv_per_wins= [-1, -1, -1, -1],
                 topks= [1, 4, 16, -2],
                 side_dwconv= 5,
                 before_attn_dwconv= 3,
                 layer_scale_init_value= -1,
                 qk_dims= [64, 128, 256, 512],
                 head_dim= 32,
                 param_routing= False,
                 diff_routing= False,
                 soft_routing= False,
                 pre_norm= True,
                 pe= None,
                 auto_pad= True,
                 use_checkpoint_stages= [], 
                 drop_path_rate= 0.15):
        super().__init__(depth=depth,
                         in_chans=in_chans,
                         embed_dim=embed_dim,
                         num_classes=num_classes,
                         mlp_ratios=mlp_ratios,
                         n_win=n_win,
                         kv_downsample_mode=kv_downsample_mode,
                         kv_per_wins=kv_per_wins,
                         topks=topks,
                         side_dwconv=side_dwconv,
                         before_attn_dwconv=before_attn_dwconv,
                         layer_scale_init_value=layer_scale_init_value,
                         qk_dims=qk_dims,
                         head_dim=head_dim,
                         param_routing=param_routing,
                         diff_routing=diff_routing,
                         soft_routing=soft_routing,
                         pre_norm=pre_norm,
                         pe=pe,
                         auto_pad=auto_pad,
                         use_checkpoint_stages=use_checkpoint_stages,
                         drop_path_rate=drop_path_rate)
        
        # step 1: remove unused segmentation head & norm
        del self.head # classification head
        del self.norm # head norm

        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        for i in range(len(embed_dim)): #4
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
        
        # step 3: initialization & load ckpt
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)

        # step 4: convert sync bn, as the batch size is too small in segmentation
        # TODO: check if this is correct
        #nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
           # logger = get_root_logger()
           # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            print(f'Load pretrained model from {pretrained}')   
    
    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(len(self.embed_dim)): # 4
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # DONE: check the inconsistency -> no effect on performance
            # in the version before submission:
            # x = self.extra_norms[i](x)
            # out.append(x)
            out.append(self.extra_norms[i](x))
        return tuple(out)
    
    def forward(self, x:torch.Tensor):
        return self.forward_features(x) #TODO: we return a tuple here. Maybe change to list


##############################################
#
# BiFormer Encoder DAEFormer Decoder
#
##############################################

class BiDAEFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip", topks=[1, 16, -2]):
        super().__init__()

        # Encoder
        """
        dims, key_dim, value_dim, layers = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2]]
        self.backbone = MiT(
            image_size=224,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )
        """
        self.backbone = BiFormer_mm(depth= [4, 18, 4],
                                    in_chans=3,
                                    num_classes=num_classes,
                                    embed_dim= [128, 320, 512],
                                    mlp_ratios= [3, 3, 3],
                                    n_win= 8,
                                    kv_downsample_mode= 'identity',
                                    kv_per_wins= [-1, -1, -1],
                                    topks= topks, #[1, 16, -2], # [1, 4, 16, -2]
                                    side_dwconv= 5,
                                    before_attn_dwconv= 3,
                                    layer_scale_init_value= -1,
                                    qk_dims= [128, 320, 512],
                                    head_dim= 32,
                                    param_routing= False,
                                    diff_routing= False,
                                    soft_routing= False,
                                    pre_norm= True,
                                    pe= None,
                                    auto_pad= True,
                                    use_checkpoint_stages= [],
                                    drop_path_rate= 0.15)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
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

        b, c, _, _ = output_enc[2].shape

        # ---------------Decoder-------------------------
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))

        return tmp_0



if __name__ == '__main__':

    DEVICE = 'cuda'
    input = torch.rand((1,1,224,224)).to(DEVICE)

    model = BiDAEFormer()
    model.to(DEVICE)
    output = model(input)
    print("Output shape: {}".format(output.size()))
    print("Finished test.")

    