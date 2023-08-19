# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


"""MViT models."""

import math
from functools import partial

import torch
import torch.nn as nn
from mvit_attention import MultiScaleBlock
from mvit_common import round_width
#from mvit.utils.misc import validate_checkpoint_wrapper_import
from torch.nn.init import trunc_normal_

#from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape


class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


#@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, #cfg, 
                 spatial_size=224,
                 embed_dim=96,
                 num_heads=1, 
                 depth=16,
                 cls_embed_on=False,
                 num_classes=1000,
                 use_abs_pos=False,
                 zero_decay_pos_cls=False,
                 patch_kernel=[7,7],
                 patch_stride=[3,3],
                 patch_padding=[3,3],
                 drop_path_rate=0.1,
                 dim_mul_in_att=True,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 mvit_mode="conv",
                 pool_first=False,
                 rel_pos_spatial=True,
                 rel_pos_zero_init=False,
                 residual_pooling=True,
                 dropout_rate=0.0,
                 head_act="softmax",
                 cfg_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]], #[],
                 cfg_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]], #[],
                 pool_q_stride=[[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1], [12, 1, 1], [13, 1, 1], [14, 2, 2], [15, 1, 1]],
                 pool_kvq_kernel=(3,3),
                 pool_kv_stride=None,
                 pool_kv_stride_adaptive=[4, 4], #None,
                 ):
        super().__init__()
        # Get parameters.
        #assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        # Prepare input.
        in_chans = 3
        spatial_size = 224 #cfg.DATA.TRAIN_CROP_SIZE
        # Prepare output.
        num_classes = num_classes #cfg.MODEL.NUM_CLASSES
        embed_dim = embed_dim #cfg.MVIT.EMBED_DIM
        # MViT params.
        num_heads = num_heads #cfg.MVIT.NUM_HEADS
        depth = depth #cfg.MVIT.DEPTH
        self.cls_embed_on = cls_embed_on, #cfg.MVIT.CLS_EMBED_ON
        self.use_abs_pos = use_abs_pos, #cfg.MVIT.USE_ABS_POS
        self.zero_decay_pos_cls = zero_decay_pos_cls, #cfg.MVIT.ZERO_DECAY_POS_CLS

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        #if cfg.MODEL.ACT_CHECKPOINT:
        #    validate_checkpoint_wrapper_import(checkpoint_wrapper)

        patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=patch_kernel, #cfg.MVIT.PATCH_KERNEL,
            stride=patch_stride, #cfg.MVIT.PATCH_STRIDE,
            padding=patch_padding, #cfg.MVIT.PATCH_PADDING,
        )
        #if cfg.MODEL.ACT_CHECKPOINT:
        #    patch_embed = checkpoint_wrapper(patch_embed)
        self.patch_embed = patch_embed

        patch_dims = [
            spatial_size // patch_stride[0], #cfg.MVIT.PATCH_STRIDE[0],
            spatial_size // patch_stride[1], #cfg.MVIT.PATCH_STRIDE[1],
        ]
        num_patches = math.prod(patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        # MViT backbone configs
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = _prepare_mvit_configs(
            cfg_depth=depth,
            cfg_head_mul=cfg_head_mul,
            cfg_dim_mul=cfg_dim_mul,
            pool_kv_stride=pool_kv_stride,
            pool_kvq_kernel=pool_kvq_kernel,
            pool_kv_stride_adaptive=pool_kv_stride_adaptive,
            pool_q_stride=pool_q_stride,
        )

        input_size = patch_dims
        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if dim_mul_in_att: #cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio, #cfg.MVIT.MLP_RATIO,
                qkv_bias=qkv_bias, #cfg.MVIT.QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mvit_mode, #cfg.MVIT.MODE,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first, #cfg.MVIT.POOL_FIRST,
                rel_pos_spatial=rel_pos_spatial, #cfg.MVIT.REL_POS_SPATIAL,
                rel_pos_zero_init=rel_pos_zero_init, #cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=residual_pooling, #cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=dim_mul_in_att, #cfg.MVIT.DIM_MUL_IN_ATT,
            )

            #if cfg.MODEL.ACT_CHECKPOINT:
            #    attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)

        self.head = TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=dropout_rate,#cfg.MODEL.DROPOUT_RATE,
            act_func=head_act, #cfg.MODEL.HEAD_ACT,
        )
        if self.use_abs_pos:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            # add all potential params
            names = ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"]

        return names

    def forward(self, x):
        x, bchw = self.patch_embed(x)

        H, W = bchw[-2], bchw[-1]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            x = x + self.pos_embed

        thw = [H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)

        if self.cls_embed_on:
            x = x[:, 0]
        else:
            x = x.mean(1)

        x = self.head(x)
        return x


def _prepare_mvit_configs(#cfg,
                          cfg_depth,
                          cfg_dim_mul,
                          cfg_head_mul,
                          pool_q_stride,
                          pool_kvq_kernel,
                          pool_kv_stride,
                          pool_kv_stride_adaptive):
    """
    Prepare mvit configs for dim_mul and head_mul facotrs, and q and kv pooling
    kernels and strides.
    """
    depth = cfg_depth #cfg.MVIT.DEPTH
    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    for i in range(len(cfg_dim_mul)): # cfg.MVIT.DIM_MUL
        dim_mul[cfg_dim_mul[i][0]] = cfg_dim_mul[i][1]
    for i in range(len(cfg_head_mul)):
        head_mul[cfg_head_mul[i][0]] = cfg_head_mul[i][1]

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

    for i in range(len(pool_q_stride)):
        stride_q[pool_q_stride[i][0]] = pool_q_stride[i][1:]
        pool_q[pool_q_stride[i][0]] = pool_kvq_kernel

    # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
    if pool_kv_stride_adaptive is not None:
        _stride_kv = pool_kv_stride_adaptive
        pool_kv_stride = []
        for i in range(cfg_depth):
            if len(stride_q[i]) > 0:
                _stride_kv = [
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            pool_kv_stride.append([i] + _stride_kv)

    for i in range(len(pool_kv_stride)):
        stride_kv[pool_kv_stride[i][0]] = pool_kv_stride[i][1:]
        pool_kv[pool_kv_stride[i][0]] = pool_kvq_kernel

    return dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv


if __name__ == "__main__":
    input = torch.rand((1,3,224,224)).cuda(0)
    net = MViT().cuda(0)
    output = net(input)

