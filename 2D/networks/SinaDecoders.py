import numpy as np
import torch
import torch.nn as nn
import einops
import math
from timm.models.layers import to_2tuple

from DeformableAttention import DAttentionBaseline, LayerNormProxy, LocalAttention, TransformerMLPWithConv, ShiftWindowAttention
from segformer import MixFFN,MixFFN_skip, MLP_FFN
from MaxViT_deform_LKA import PatchExpand, FinalPatchExpand_X4

#from .DeformableAttention import DAttentionBaseline, LayerNormProxy, LocalAttention, TransformerMLPWithConv, ShiftWindowAttention
#from .segformer import MixFFN,MixFFN_skip, MLP_FFN
#from .MaxViT_deform_LKA import PatchExpand, FinalPatchExpand_X4

class DATMixFFNTransformerBlock(nn.Module):
    def __init__(self, dim, token_mlp, input_size, heads = 24, hc = 32, n_groups= 6,
               attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1, use_pe = False,
               dwc_pe = True, no_off = False, fixed_pe= False, ksize = 9, log_cpb = False):
      super().__init__()
    #َArguments
      self.dim = dim
      self.token_mlp = token_mlp
      self.img_size = to_2tuple(input_size)
      self.heads = heads
      self.hc = hc
      self.n_groups = n_groups
      self.attn_drop = attn_drop
      self.proj_drop = proj_drop
      self.stride = stride
      self.offset_range_factor = offset_range_factor
      self.use_pe = use_pe
      self.dwc_pe = dwc_pe
      self.no_off = no_off
      self.fixed_pe = fixed_pe
      self.ksize = ksize
      self.log_cpb = log_cpb

      hc = dim // heads
      assert dim == heads * hc


    #Layers
      self.layernorm1 = LayerNormProxy(dim)
      self.deformattn = DAttentionBaseline(self.img_size,self.img_size,self.heads,self.hc,self.n_groups,
                                         self.attn_drop,self.proj_drop,self.stride,self.offset_range_factor,
                                         self.use_pe, self.dwc_pe, self.no_off, self.fixed_pe, self.ksize, self.log_cpb)
      self.layernorm2 = LayerNormProxy(dim)
      self.mlp = MixFFN_skip(dim,dim)
      if token_mlp=='mix':
            self.mlp = MixFFN(dim, int(dim*4))
      elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim*4))
      else:
            self.mlp = MLP_FFN(dim, int(dim*4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x1= self.layernorm1(x)
        x1, pos, ref = self.deformattn(x1)

        x1 = x1 + x_res
        x_res = x1

        N = x1.size(2) * x1.size(3) # N = H * W
        x1 = self.layernorm2(x1).contiguous()
        x1 = x1.view(x1.size(0), N, x1.size(1)) # Reshape to [B, N, C]
        x1 = self.mlp(x1, x_res.size(2), x_res.size(3)) # MIX-FFN (MissFormer)

        out = x1.view(x_res.size(0), x_res.size(1), x_res.size(2) , x_res.size(3)) + x_res #Reshape to [B, C, H , W]

        return out


class LOCMixFFNTransformerBlock(nn.Module):
    def __init__(self, dim, token_mlp, window_size, input_size, heads=24, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        # َArguments (DAT)
        self.dim = dim
        self.token_mlp = token_mlp
        self.window_size = window_size
        self.input_size = input_size
        self.img_size = to_2tuple(input_size)
        self.heads = heads
        # self.hc = hc
        # self.n_groups = n_groups
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        # self.stride = stride
        # self.offset_range_factor = offset_range_factor
        # self.use_pe = use_pe
        # self.dwc_pe = dwc_pe
        # self.no_off = no_off
        # self.fixed_pe = fixed_pe
        # self.ksize = ksize
        # self.log_cpb = log_cpb

        # hc = dim // heads
        # assert dim == heads * hc
        # Arguments (LOC)

        # Layers
        self.layernorm1 = LayerNormProxy(dim)
        # self.deformattn = DAttentionBaseline(self.img_size,self.img_size,self.heads,self.hc,self.n_groups,
        #                                   self.attn_drop,self.proj_drop,self.stride,self.offset_range_factor,
        #                                   self.use_pe, self.dwc_pe, self.no_off, self.fixed_pe, self.ksize, self.log_cpb)

        self.localattn = LocalAttention(dim, self.heads, self.window_size, self.attn_drop, self.proj_drop)
        self.layernorm2 = LayerNormProxy(dim)
        self.mlp = MixFFN_skip(dim, dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x1 = self.layernorm1(x)
        x1, pos, ref = self.localattn(x1)

        x1 = x1 + x_res
        x_res = x1

        N = x1.size(2) * x1.size(3)  # N = H * W
        x1 = self.layernorm2(x1).contiguous()
        x1 = x1.view(x1.size(0), N, x1.size(1))  # Reshape to [B, N, C]
        x1 = self.mlp(x1, x_res.size(2), x_res.size(3))  # MIX-FFN (MissFormer)

        out = x1.view(x_res.size(0), x_res.size(1), x_res.size(2), x_res.size(3)) + x_res  # Reshape to [B, C, H , W]

        return out

class SWMixFFNTransformerBlock(nn.Module):
    def __init__(self, dim, token_mlp, window_size, input_size, heads=24, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        # َArguments (DAT)
        self.dim = dim
        self.token_mlp = token_mlp
        self.window_size = window_size
        self.input_size = input_size
        self.img_size = to_2tuple(input_size)
        self.heads = heads
        # self.hc = hc
        # self.n_groups = n_groups
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.shift_size = math.ceil(window_size / 2)
        # self.stride = stride
        # self.offset_range_factor = offset_range_factor
        # self.use_pe = use_pe
        # self.dwc_pe = dwc_pe
        # self.no_off = no_off
        # self.fixed_pe = fixed_pe
        # self.ksize = ksize
        # self.log_cpb = log_cpb

        # hc = dim // heads
        # assert dim == heads * hc
        # Arguments (LOC)

        # Layers
        self.layernorm1 = LayerNormProxy(dim)
        # self.deformattn = DAttentionBaseline(self.img_size,self.img_size,self.heads,self.hc,self.n_groups,
        #                                   self.attn_drop,self.proj_drop,self.stride,self.offset_range_factor,
        #                                   self.use_pe, self.dwc_pe, self.no_off, self.fixed_pe, self.ksize, self.log_cpb)

        self.swattn = ShiftWindowAttention(dim = dim, heads=self.heads, window_size=self.window_size,
                                           attn_drop=self.attn_drop, shift_size=self.shift_size,
                                           proj_drop=self.proj_drop, fmap_size= self.input_size)
        self.layernorm2 = LayerNormProxy(dim)
        self.mlp = MixFFN_skip(dim, dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x1 = self.layernorm1(x)
        x1, pos, ref = self.swattn(x1)

        x1 = x1 + x_res
        x_res = x1

        N = x1.size(2) * x1.size(3)  # N = H * W
        x1 = self.layernorm2(x1).contiguous()
        x1 = x1.view(x1.size(0), N, x1.size(1))  # Reshape to [B, N, C]
        x1 = self.mlp(x1, x_res.size(2), x_res.size(3))  # MIX-FFN (MissFormer)

        out = x1.view(x_res.size(0), x_res.size(1), x_res.size(2), x_res.size(3)) + x_res  # Reshape to [B, C, H , W]

        return out

class MyDecoderLayer1(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, n_groups, n_class=3, norm_layer=nn.LayerNorm, is_last=False, is_first= False):
        super().__init__()
        #dims = in_out_chan[0]
        dims , out_dim = in_out_chan
        if is_first:
            self.concat_linear = nn.Linear(dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=dims, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None

        elif not is_last and not is_first:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0], token_mlp = 'mix_skip', heads = heads, hc = 32, n_groups= n_groups,
                                                        attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
                                                         use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
                                                         ksize = 9, log_cpb = False)
        self.layer_former_2 = DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0],token_mlp = 'mix_skip',  heads = heads, hc = 32, n_groups= n_groups,
                                                        attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
                                                         use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
                                                         ksize = 9, log_cpb = False)


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
        if x2 is not None:
            b, c, h, w = x1.shape
            b2, c2, h2, w2 = x2.shape
            x2 = x2.view(b2, -1, c2)  #B*N*C #### From Skip connection
            x1 = x1.view(b, -1, c)  #B*N*C  #### From Last Decoder Layer


            #print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)  # B*N*2C
            #print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)  #B*N*C
            #print("-----cat_linear_x shape", cat_linear_x.shape)
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
            #print("-----cat_linear_x shaped shape", cat_linear_x.shape)
            tran_layer_1 = self.layer_former_1(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2), tran_layer_2.size(1))

            if self.last_layer:

                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2))
            else:

                out = self.layer_up(tran_layer_2)
                out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
                #out = out.view(out.size(0), out.size(2), out.size(1)//h,  out.size(1)//w)
                #out = out.view(tran_layer_2.size(0), tran_layer_2.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
        else:
          b, c, h , w = x1.shape
          x1 = x1.view(x1.size(0) , -1, x1.size(1))
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
          out = self.layer_up(x1)
          out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
        return out
class MyDecoderLayer2(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, n_groups, window_size, n_class=3, norm_layer=nn.LayerNorm, is_last=False, is_first= False):
        super().__init__()
        #dims = in_out_chan[0]
        dims , out_dim = in_out_chan
        if is_first:
            self.concat_linear = nn.Linear(dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=dims, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None

        elif not is_last and not is_first:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        #self.layer_former_1 =  DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0], token_mlp = 'mix_skip', heads = heads, hc = 32, n_groups= n_groups,
        #                                                attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
        #                                                 use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
        #                                                 ksize = 9, log_cpb = False)
        self.layer_former_1 = LOCMixFFNTransformerBlock(dim = out_dim, token_mlp= 'mix_skip', input_size = input_size[0], window_size = window_size, heads = heads, attn_drop = 0.0, proj_drop = 0.0)

        self.layer_former_2 = DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0],token_mlp = 'mix_skip',  heads = heads, hc = 32, n_groups= n_groups,
                                                        attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
                                                         use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
                                                         ksize = 9, log_cpb = False)


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
        if x2 is not None:
            b, c, h, w = x1.shape
            b2, c2, h2, w2 = x2.shape
            x2 = x2.view(b2, -1, c2)  #B*N*C #### From Skip connection
            x1 = x1.view(b, -1, c)  #B*N*C  #### From Last Decoder Layer


            #print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)  # B*N*2C
            #print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)  #B*N*C
            #print("-----cat_linear_x shape", cat_linear_x.shape)
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
            #print("-----cat_linear_x shaped shape", cat_linear_x.shape)
            tran_layer_1 = self.layer_former_1(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2), tran_layer_2.size(1))

            if self.last_layer:

                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2))
            else:

                out = self.layer_up(tran_layer_2)
                out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
                #out = out.view(out.size(0), out.size(2), out.size(1)//h,  out.size(1)//w)
                #out = out.view(tran_layer_2.size(0), tran_layer_2.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
        else:
          b, c, h , w = x1.shape
          x1 = x1.view(x1.size(0) , -1, x1.size(1))
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
          out = self.layer_up(x1)
          out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
        return out

class MyDecoderLayer3(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, n_groups, window_size, n_class=3, norm_layer=nn.LayerNorm, is_last=False, is_first= False):
        super().__init__()
        #dims = in_out_chan[0]
        dims , out_dim = in_out_chan
        if is_first:
            self.concat_linear = nn.Linear(dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=dims, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None

        elif not is_last and not is_first:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        #self.layer_former_1 =  DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0], token_mlp = 'mix_skip', heads = heads, hc = 32, n_groups= n_groups,
        #                                                attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
        #                                                 use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
        #                                                 ksize = 9, log_cpb = False)
        self.layer_former_1 = LOCMixFFNTransformerBlock(dim=out_dim, token_mlp='mix_skip', input_size=input_size[0],
                                                        window_size = window_size, heads=heads, attn_drop=0.0,
                                                        proj_drop=0.0)
        self.layer_former_2 = LOCMixFFNTransformerBlock(dim=out_dim, token_mlp='mix_skip', input_size=input_size[0],
                                                        window_size=window_size, heads=heads, attn_drop=0.0,
                                                        proj_drop=0.0)
        #self.layer_former_2 = DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0],token_mlp = 'mix_skip',  heads = heads, hc = 32, n_groups= n_groups,
        #                                                attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
        #                                                 use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
        #                                                 ksize = 9, log_cpb = False)


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
        if x2 is not None:
            b, c, h, w = x1.shape
            b2, c2, h2, w2 = x2.shape
            x2 = x2.view(b2, -1, c2)  #B*N*C #### From Skip connection
            x1 = x1.view(b, -1, c)  #B*N*C  #### From Last Decoder Layer


            #print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)  # B*N*2C
            #print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)  #B*N*C
            #print("-----cat_linear_x shape", cat_linear_x.shape)
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
            #print("-----cat_linear_x shaped shape", cat_linear_x.shape)
            tran_layer_1 = self.layer_former_1(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2), tran_layer_2.size(1))

            if self.last_layer:

                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2))
            else:

                out = self.layer_up(tran_layer_2)
                out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
                #out = out.view(out.size(0), out.size(2), out.size(1)//h,  out.size(1)//w)
                #out = out.view(tran_layer_2.size(0), tran_layer_2.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
        else:
          b, c, h , w = x1.shape
          x1 = x1.view(x1.size(0) , -1, x1.size(1))
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
          out = self.layer_up(x1)
          out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
        return out
class MyDecoderLayer2Deep(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, n_groups, window_size, n_class=3, norm_layer=nn.LayerNorm, is_last=False, is_first= False):
        super().__init__()
        #dims = in_out_chan[0]
        dims , out_dim = in_out_chan
        if is_first:
            self.concat_linear = nn.Linear(dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=dims, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None

        elif not is_last and not is_first:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        #self.layer_former_1 =  DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0], token_mlp = 'mix_skip', heads = heads, hc = 32, n_groups= n_groups,
        #                                                attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
        #                                                 use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
        #                                                 ksize = 9, log_cpb = False)
        self.layer_former_1 = LOCMixFFNTransformerBlock(dim = out_dim, token_mlp= 'mix_skip', input_size = input_size[0], window_size = window_size, heads = heads, attn_drop = 0.0, proj_drop = 0.0)

        self.layer_former_2 = DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0],token_mlp = 'mix_skip',  heads = heads, hc = 32, n_groups= n_groups,
                                                        attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
                                                         use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
                                                         ksize = 9, log_cpb = False)
        self.layer_former_3 = LOCMixFFNTransformerBlock(dim=out_dim, token_mlp='mix_skip', input_size=input_size[0],
                                                        window_size=window_size, heads=heads, attn_drop=0.0,
                                                        proj_drop=0.0)

        self.layer_former_4 = DATMixFFNTransformerBlock(dim=out_dim, input_size=input_size[0], token_mlp='mix_skip',
                                                        heads=heads, hc=32, n_groups=n_groups,
                                                        attn_drop=0.0, proj_drop=0.0, stride=1, offset_range_factor=1,
                                                        use_pe=False, dwc_pe=True, no_off=False, fixed_pe=False,
                                                        ksize=9, log_cpb=False)
        self.layer_former_5 = LOCMixFFNTransformerBlock(dim=out_dim, token_mlp='mix_skip', input_size=input_size[0],
                                                        window_size=window_size, heads=heads, attn_drop=0.0,
                                                        proj_drop=0.0)

        self.layer_former_6 = DATMixFFNTransformerBlock(dim=out_dim, input_size=input_size[0], token_mlp='mix_skip',
                                                        heads=heads, hc=32, n_groups=n_groups,
                                                        attn_drop=0.0, proj_drop=0.0, stride=1, offset_range_factor=1,
                                                        use_pe=False, dwc_pe=True, no_off=False, fixed_pe=False,
                                                        ksize=9, log_cpb=False)

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
        if x2 is not None:
            b, c, h, w = x1.shape
            b2, c2, h2, w2 = x2.shape
            x2 = x2.view(b2, -1, c2)  #B*N*C #### From Skip connection
            x1 = x1.view(b, -1, c)  #B*N*C  #### From Last Decoder Layer


            #print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)  # B*N*2C
            #print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)  #B*N*C
            #print("-----cat_linear_x shape", cat_linear_x.shape)
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
            #print("-----cat_linear_x shaped shape", cat_linear_x.shape)
            tran_layer_1 = self.layer_former_1(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)
            tran_layer_3 = self.layer_former_3(tran_layer_2)
            tran_layer_4 = self.layer_former_4(tran_layer_3)
            tran_layer_5 = self.layer_former_5(tran_layer_4)
            tran_layer_6 = self.layer_former_6(tran_layer_5)

            tran_layer_6 = tran_layer_6.view(tran_layer_6.size(0), tran_layer_6.size(3) * tran_layer_6.size(2), tran_layer_6.size(1))

            if self.last_layer:

                out = self.last_layer(self.layer_up(tran_layer_6).view(b, 4*h, 4*w, -1).permute(0,3,1,2))
            else:

                out = self.layer_up(tran_layer_6)
                out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
                #out = out.view(out.size(0), out.size(2), out.size(1)//h,  out.size(1)//w)
                #out = out.view(tran_layer_2.size(0), tran_layer_2.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
        else:
          b, c, h , w = x1.shape
          x1 = x1.view(x1.size(0) , -1, x1.size(1))
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
          out = self.layer_up(x1)
          out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
        return out

class MyDecoderLayer4(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, n_groups, window_size, n_class=3, norm_layer=nn.LayerNorm, is_last=False, is_first= False):
        super().__init__()
        #dims = in_out_chan[0]
        dims , out_dim = in_out_chan
        if is_first:
            self.concat_linear = nn.Linear(dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=dims, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None

        elif not is_last and not is_first:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        #self.layer_former_1 =  DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0], token_mlp = 'mix_skip', heads = heads, hc = 32, n_groups= n_groups,
        #                                                attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
        #                                                 use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
        #                                                 ksize = 9, log_cpb = False)
        self.layer_former_1 = LOCMixFFNTransformerBlock(dim=out_dim, token_mlp='mix_skip', input_size=input_size[0],
                                                        window_size = window_size, heads=heads, attn_drop=0.0,
                                                        proj_drop=0.0)
        self.layer_former_2 = SWMixFFNTransformerBlock(dim = out_dim, token_mlp='mix_skip', window_size=window_size,
                                                       input_size=input_size[0], heads=heads, attn_drop=0.0,
                                                       proj_drop=0.0)

        #self.layer_former_2 = LOCMixFFNTransformerBlock(dim=out_dim, token_mlp='mix_skip', input_size=input_size[0],
        #                                                window_size=window_size, heads=heads, attn_drop=0.0,
        #                                                proj_drop=0.0)
        #self.layer_former_2 = DATMixFFNTransformerBlock(dim = out_dim, input_size = input_size[0],token_mlp = 'mix_skip',  heads = heads, hc = 32, n_groups= n_groups,
        #                                                attn_drop = 0.0, proj_drop = 0.0, stride = 1, offset_range_factor = 1,
        #                                                 use_pe = False, dwc_pe = True, no_off = False, fixed_pe= False,
        #                                                 ksize = 9, log_cpb = False)


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
        if x2 is not None:
            b, c, h, w = x1.shape
            b2, c2, h2, w2 = x2.shape
            x2 = x2.view(b2, -1, c2)  #B*N*C #### From Skip connection
            x1 = x1.view(b, -1, c)  #B*N*C  #### From Last Decoder Layer


            #print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)  # B*N*2C
            #print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)  #B*N*C
            #print("-----cat_linear_x shape", cat_linear_x.shape)
            #tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            #tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            cat_linear_x = cat_linear_x.view(cat_linear_x.size(0), cat_linear_x.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
            #print("-----cat_linear_x shaped shape", cat_linear_x.shape)
            tran_layer_1 = self.layer_former_1(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2), tran_layer_2.size(1))

            if self.last_layer:

                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2))
            else:

                out = self.layer_up(tran_layer_2)
                out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
                #out = out.view(out.size(0), out.size(2), out.size(1)//h,  out.size(1)//w)
                #out = out.view(tran_layer_2.size(0), tran_layer_2.size(2), cat_linear_x.size(1)//w, cat_linear_x.size(1)//h)
        else:
          b, c, h , w = x1.shape
          x1 = x1.view(x1.size(0) , -1, x1.size(1))
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
          out = self.layer_up(x1)
          out = out.view(out.size(0),  out.size(2), out.size(1)//(h*2), out.size(1)//(w*2))
        return out

class MISSFormer3(nn.Module):
    def __init__(self, num_classes=9, encoder_pretrained=True):
        super().__init__()

        #reduction_ratios = [8, 4, 2, 1]
        #heads = [1, 2, 5, 8]
        heads = [3, 6, 12, 24]
        n_groups = [48,24,12,6]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        #in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        in_out_chan = [[96, 96],[192, 192],[384, 384],[768, 768]]

        #dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]

        self.backbone = DATMiss2(img_size=56, patch_size=4, expansion=4,dim_stem = 96,
                                dims=[96, 192, 384, 768], depths=[2,2,2,2],
                                heads=[3,6,12,24], window_sizes=[7,7,7,7],
                                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                                strides=[1,1,1,1], offset_range_factor=[1,2,3,4],
                                stage_spec=[['DM',"DM"], ['DM',"DM"],['DM',"DM"],['DM',"DM"]],
                                groups=[1,1,3,6], use_pes=[False,False,False,False],
                                dwc_pes=[False,False,False,False], sr_ratios=[8, 4, 2, 1],
                                lower_lr_kvs={}, fixed_pes=[False,False,False,False],
                                no_offs=[False,False,False,False], ns_per_pts=[4,4,4,4],
                                use_dwc_mlps=[False,False,False,False], use_conv_patches= True,
                                ksizes=[9, 7, 5, 3] ,ksize_qnas=[3, 3, 3, 3], nqs=[2, 2, 2, 2],
                                qna_activation='exp', nat_ksizes=[3,3,3,3], layer_scale_values=[-1,-1,-1,-1],
                                use_lpus=[False,False,False,False], log_cpb=[False,False,False,False])

        #self.reduction_ratios = [1, 2, 4, 8]
        #self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)


        #dec_4 = MyDecoderLayer(input_size = (7,7), in_out_chan= (768,768), heads = 24, n_groups = 48, is_first = True)
        #dec_3 = MyDecoderLayer(input_size = (14,14), in_out_chan= (384,384), heads = 12, n_groups = 24)
        #ec_2 = MyDecoderLayer(input_size = (28,28), in_out_chan= (192,192), heads = 6, n_groups = 12)
        #dec_1 = MyDecoderLayer(input_size = (56,56), in_out_chan= (96,96), heads = 3, n_groups = 6, is_last = True)


        self.decoder_4= MyDecoderLayer1((d_base_feat_size,d_base_feat_size), in_out_chan[3], heads[3], n_groups[0],n_class=num_classes, is_first= True)
        self.decoder_3= MyDecoderLayer1((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], n_class=num_classes)
        self.decoder_2= MyDecoderLayer1((d_base_feat_size*4,d_base_feat_size*4), in_out_chan[1], heads[1], n_groups[2], n_class=num_classes)
        #self.decoder_1= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], n_class=num_classes, is_last=True)
        self.decoder_1 = MyDecoderLayer1((14, 14), in_out_chan[0], heads[0], n_groups[3], n_class=num_classes, is_last=True)

    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        bottleneck, skip_inputs, _, _ = self.backbone(x)
        # bridge = self.bridge(encoder) #list

        # b,c,_,_ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        # ---------------Decoder-------------------------
        # print("stage3-----")
        # tmp_3 = self.decoder_3(bridge[3].permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")
        # tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0,2,3,1))
        # print("stage1-----")
        # tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0,2,3,1))
        # print("stage0-----")
        # tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0,2,3,1))

        # modified decoder
        # x = self.decoder_4(x1 = bottleneck, x2= None)

        # x = self.decoder_3(x1 = x ,x2= skip_inputs[2])

        # x = dec3(x , inp1_1)

        # x = self.decoder_2(x1 = x ,x2= skip_inputs[1])

        # x = dec4(x , inp0_1)
        # x = self.decoder_1(x1 = x ,x2= skip_inputs[0])
        x = self.decoder_1(x1=bottleneck, x2=skip_inputs[0])

        return x, skip_inputs[0]
class MISSFormer4(nn.Module):
    def __init__(self, num_classes=9, encoder_pretrained=True):
        super().__init__()

        #reduction_ratios = [8, 4, 2, 1]
        #heads = [1, 2, 5, 8]
        window_sizes=[7,7,7,7]
        heads = [3, 6, 12, 24]
        n_groups = [48,24,12,6]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        #in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        in_out_chan = [[96, 96],[192, 192],[384, 384],[768, 768]]

        #dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]

        self.backbone = DATMissLG(img_size=56, patch_size=4, expansion=4,dim_stem = 96,
                                dims=[96, 192, 384, 768], depths=[2,2,2,2],
                                heads=[3,6,12,24], window_sizes=[7,7,7,7],
                                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                                strides=[1,1,1,1], offset_range_factor=[1,2,3,4],
                                stage_spec=[['L',"DM"], ['L',"DM"],['L',"DM"],['DM',"DM"]],
                                groups=[1,1,3,6], use_pes=[False,False,False,False],
                                dwc_pes=[False,False,False,False], sr_ratios=[8, 4, 2, 1],
                                lower_lr_kvs={}, fixed_pes=[False,False,False,False],
                                no_offs=[False,False,False,False], ns_per_pts=[4,4,4,4],
                                use_dwc_mlps=[False,False,False,False], use_conv_patches= True,
                                ksizes=[9, 7, 5, 3] ,ksize_qnas=[3, 3, 3, 3], nqs=[2, 2, 2, 2],
                                qna_activation='exp', nat_ksizes=[3,3,3,3], layer_scale_values=[-1,-1,-1,-1],
                                use_lpus=[False,False,False,False], log_cpb=[False,False,False,False])

        #self.reduction_ratios = [1, 2, 4, 8]
        #self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)


        #dec_4 = MyDecoderLayer(input_size = (7,7), in_out_chan= (768,768), heads = 24, n_groups = 48, is_first = True)
        #dec_3 = MyDecoderLayer(input_size = (14,14), in_out_chan= (384,384), heads = 12, n_groups = 24)
        #ec_2 = MyDecoderLayer(input_size = (28,28), in_out_chan= (192,192), heads = 6, n_groups = 12)
        #dec_1 = MyDecoderLayer(input_size = (56,56), in_out_chan= (96,96), heads = 3, n_groups = 6, is_last = True)


        self.decoder_4= MyDecoderLayer1((d_base_feat_size,d_base_feat_size), in_out_chan[3], heads[3], n_groups[0],n_class=num_classes, is_first= True)
        self.decoder_3= MyDecoderLayer1((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], n_class=num_classes)
        self.decoder_2= MyDecoderLayer2((d_base_feat_size*4,d_base_feat_size*4), in_out_chan[1], heads[1], n_groups[2], window_size= window_sizes[0], n_class=num_classes)
        #self.decoder_1= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], n_class=num_classes, is_last=True)
        self.decoder_1= MyDecoderLayer2((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], window_size= window_sizes[0], n_class=num_classes, is_last=True)

    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        bottleneck, skip_inputs, _, _ = self.backbone(x)
        #bridge = self.bridge(encoder) #list

        #b,c,_,_ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        #---------------Decoder-------------------------
        # print("stage3-----")
        #tmp_3 = self.decoder_3(bridge[3].permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")
        #tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0,2,3,1))
        # print("stage1-----")
        #tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0,2,3,1))
        # print("stage0-----")
        #tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0,2,3,1))

        #modified decoder
        x = self.decoder_4(x1 = bottleneck, x2= None)

        x = self.decoder_3(x1 = x ,x2= skip_inputs[2])

        #x = dec3(x , inp1_1)

        x = self.decoder_2(x1 = x ,x2= skip_inputs[1])

        #x = dec4(x , inp0_1)
        x = self.decoder_1(x1 = x ,x2= skip_inputs[0])
        #x = self.decoder_1(x1 = bottleneck ,x2= skip_inputs[0])

        return x

class MISSFormer5(nn.Module):
    def __init__(self, num_classes=9, encoder_pretrained=True):
        super().__init__()

        #reduction_ratios = [8, 4, 2, 1]
        #heads = [1, 2, 5, 8]
        window_sizes=[7,7,7,7]
        heads = [3, 6, 12, 24]
        n_groups = [48,24,12,6]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        #in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        in_out_chan = [[96, 96],[192, 192],[384, 384],[768, 768]]

        #dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]

        self.backbone = DATMissLG2(img_size=56, patch_size=4, expansion=4,dim_stem = 96,
                                dims=[96, 192, 384, 768], depths=[2,2,6,2],
                                heads=[3,6,12,24], window_sizes=[7,7,7,7],
                                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                                strides=[1,1,1,1], offset_range_factor=[1,2,3,4],
                                stage_spec=[['L',"L"], ['L',"L"],['L',"DM",'L',"DM",'L',"DM"],['DM',"DM"]],
                                groups=[1,1,3,6], use_pes=[False,False,False,False],
                                dwc_pes=[False,False,False,False], sr_ratios=[8, 4, 2, 1],
                                lower_lr_kvs={}, fixed_pes=[False,False,False,False],
                                no_offs=[False,False,False,False], ns_per_pts=[4,4,4,4],
                                use_dwc_mlps=[False,False,False,False], use_conv_patches= True,
                                ksizes=[9, 7, 5, 3] ,ksize_qnas=[3, 3, 3, 3], nqs=[2, 2, 2, 2],
                                qna_activation='exp', nat_ksizes=[3,3,3,3], layer_scale_values=[-1,-1,-1,-1],
                                use_lpus=[False,False,False,False], log_cpb=[False,False,False,False])

        #self.reduction_ratios = [1, 2, 4, 8]
        #self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)


        #dec_4 = MyDecoderLayer(input_size = (7,7), in_out_chan= (768,768), heads = 24, n_groups = 48, is_first = True)
        #dec_3 = MyDecoderLayer(input_size = (14,14), in_out_chan= (384,384), heads = 12, n_groups = 24)
        #ec_2 = MyDecoderLayer(input_size = (28,28), in_out_chan= (192,192), heads = 6, n_groups = 12)
        #dec_1 = MyDecoderLayer(input_size = (56,56), in_out_chan= (96,96), heads = 3, n_groups = 6, is_last = True)


        self.decoder_4= MyDecoderLayer1((d_base_feat_size,d_base_feat_size), in_out_chan[3], heads[3], n_groups[0],n_class=num_classes, is_first= True)
        #self.decoder_3= MyDecoderLayer1((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], window_size= window_sizes[0], n_class=num_classes)
        self.decoder_3 = MyDecoderLayer2Deep((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], window_size= window_sizes[0], n_class=num_classes)
        self.decoder_2 = MyDecoderLayer3((d_base_feat_size*4,d_base_feat_size*4), in_out_chan[1], heads[1], n_groups[2], window_size= window_sizes[0], n_class=num_classes)
        #self.decoder_1= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], n_class=num_classes, is_last=True)
        self.decoder_1 = MyDecoderLayer3((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], window_size= window_sizes[0], n_class=num_classes, is_last=True)

    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        bottleneck, skip_inputs, _, _ = self.backbone(x)
        #bridge = self.bridge(encoder) #list

        #b,c,_,_ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        #---------------Decoder-------------------------
        # print("stage3-----")
        #tmp_3 = self.decoder_3(bridge[3].permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")
        #tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0,2,3,1))
        # print("stage1-----")
        #tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0,2,3,1))
        # print("stage0-----")
        #tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0,2,3,1))

        #modified decoder
        x = self.decoder_4(x1 = bottleneck, x2= None)

        x = self.decoder_3(x1 = x ,x2= skip_inputs[2])

        #x = dec3(x , inp1_1)

        x = self.decoder_2(x1 = x ,x2= skip_inputs[1])

        #x = dec4(x , inp0_1)
        x = self.decoder_1(x1 = x ,x2= skip_inputs[0])
        #x = self.decoder_1(x1 = bottleneck ,x2= skip_inputs[0])

        return x

class MISSFormer6(nn.Module):
    def __init__(self, num_classes=9, encoder_pretrained=True):
        super().__init__()

        #reduction_ratios = [8, 4, 2, 1]
        #heads = [1, 2, 5, 8]
        window_sizes=[7,7,7,7]
        heads = [3, 6, 12, 24]
        n_groups = [48,24,12,6]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        #in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        in_out_chan = [[96, 96],[192, 192],[384, 384],[768, 768]]

        #dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]

        self.backbone = DATMissLG2(img_size=56, patch_size=4, expansion=4,dim_stem = 96,
                                dims=[96, 192, 384, 768], depths=[2,2,6,2],
                                heads=[3,6,12,24], window_sizes=[7,7,7,7],
                                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                                strides=[1,1,1,1], offset_range_factor=[1,2,3,4],
                                stage_spec=[['L',"S"], ['L',"S"],['L',"DM",'L',"DM",'L',"DM"],['DM',"DM"]],
                                groups=[1,1,3,6], use_pes=[False,False,False,False],
                                dwc_pes=[False,False,False,False], sr_ratios=[8, 4, 2, 1],
                                lower_lr_kvs={}, fixed_pes=[False,False,False,False],
                                no_offs=[False,False,False,False], ns_per_pts=[4,4,4,4],
                                use_dwc_mlps=[False,False,False,False], use_conv_patches= True,
                                ksizes=[9, 7, 5, 3] ,ksize_qnas=[3, 3, 3, 3], nqs=[2, 2, 2, 2],
                                qna_activation='exp', nat_ksizes=[3,3,3,3], layer_scale_values=[-1,-1,-1,-1],
                                use_lpus=[False,False,False,False], log_cpb=[False,False,False,False])

        #self.reduction_ratios = [1, 2, 4, 8]
        #self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)


        #dec_4 = MyDecoderLayer(input_size = (7,7), in_out_chan= (768,768), heads = 24, n_groups = 48, is_first = True)
        #dec_3 = MyDecoderLayer(input_size = (14,14), in_out_chan= (384,384), heads = 12, n_groups = 24)
        #ec_2 = MyDecoderLayer(input_size = (28,28), in_out_chan= (192,192), heads = 6, n_groups = 12)
        #dec_1 = MyDecoderLayer(input_size = (56,56), in_out_chan= (96,96), heads = 3, n_groups = 6, is_last = True)


        self.decoder_4= MyDecoderLayer1((d_base_feat_size,d_base_feat_size), in_out_chan[3], heads[3], n_groups[0],n_class=num_classes, is_first= True)
        #self.decoder_3= MyDecoderLayer1((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], window_size= window_sizes[0], n_class=num_classes)
        self.decoder_3 = MyDecoderLayer2Deep((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], window_size= window_sizes[0], n_class=num_classes)
        self.decoder_2 = MyDecoderLayer4((d_base_feat_size*4,d_base_feat_size*4), in_out_chan[1], heads[1], n_groups[2], window_size= window_sizes[0], n_class=num_classes)
        #self.decoder_1= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], n_class=num_classes, is_last=True)
        self.decoder_1 = MyDecoderLayer4((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], window_size= window_sizes[0], n_class=num_classes, is_last=True)

    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        bottleneck, skip_inputs, _, _ = self.backbone(x)
        #bridge = self.bridge(encoder) #list

        #b,c,_,_ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        #---------------Decoder-------------------------
        # print("stage3-----")
        #tmp_3 = self.decoder_3(bridge[3].permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")
        #tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0,2,3,1))
        # print("stage1-----")
        #tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0,2,3,1))
        # print("stage0-----")
        #tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0,2,3,1))

        #modified decoder
        x = self.decoder_4(x1 = bottleneck, x2= None)

        x = self.decoder_3(x1 = x ,x2= skip_inputs[2])

        #x = dec3(x , inp1_1)

        x = self.decoder_2(x1 = x ,x2= skip_inputs[1])

        #x = dec4(x , inp0_1)
        x = self.decoder_1(x1 = x ,x2= skip_inputs[0])
        #x = self.decoder_1(x1 = bottleneck ,x2= skip_inputs[0])
        return x

class MISSFormer6(nn.Module):
    def __init__(self, num_classes=9, encoder_pretrained=True):
        super().__init__()

        #reduction_ratios = [8, 4, 2, 1]
        #heads = [1, 2, 5, 8]
        window_sizes=[7,7,7,7]
        heads = [3, 6, 12, 24]
        n_groups = [48,24,12,6]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        #in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        in_out_chan = [[96, 96],[192, 192],[384, 384],[768, 768]]

        #dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]

        self.backbone = DATMissLG2(img_size=56, patch_size=4, expansion=4,dim_stem = 96,
                                dims=[96, 192, 384, 768], depths=[2,2,6,2],
                                heads=[3,6,12,24], window_sizes=[7,7,7,7],
                                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                                strides=[1,1,1,1], offset_range_factor=[1,2,3,4],
                                stage_spec=[['E',"S"], ['E',"S"],['L',"DM",'L',"DM",'L',"DM"],['DM',"DM"]],
                                groups=[1,1,3,6], use_pes=[False,False,False,False],
                                dwc_pes=[False,False,False,False], sr_ratios=[8, 4, 2, 1],
                                lower_lr_kvs={}, fixed_pes=[False,False,False,False],
                                no_offs=[False,False,False,False], ns_per_pts=[4,4,4,4],
                                use_dwc_mlps=[False,False,False,False], use_conv_patches= True,
                                ksizes=[9, 7, 5, 3] ,ksize_qnas=[3, 3, 3, 3], nqs=[2, 2, 2, 2],
                                qna_activation='exp', nat_ksizes=[3,3,3,3], layer_scale_values=[-1,-1,-1,-1],
                                use_lpus=[False,False,False,False], log_cpb=[False,False,False,False])

        #self.reduction_ratios = [1, 2, 4, 8]
        #self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)


        #dec_4 = MyDecoderLayer(input_size = (7,7), in_out_chan= (768,768), heads = 24, n_groups = 48, is_first = True)
        #dec_3 = MyDecoderLayer(input_size = (14,14), in_out_chan= (384,384), heads = 12, n_groups = 24)
        #ec_2 = MyDecoderLayer(input_size = (28,28), in_out_chan= (192,192), heads = 6, n_groups = 12)
        #dec_1 = MyDecoderLayer(input_size = (56,56), in_out_chan= (96,96), heads = 3, n_groups = 6, is_last = True)


        self.decoder_4= MyDecoderLayer1((d_base_feat_size,d_base_feat_size), in_out_chan[3], heads[3], n_groups[0],n_class=num_classes, is_first= True)
        #self.decoder_3= MyDecoderLayer1((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], window_size= window_sizes[0], n_class=num_classes)
        self.decoder_3 = MyDecoderLayer2Deep((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], window_size= window_sizes[0], n_class=num_classes)
        self.decoder_2 = MyDecoderLayer4((d_base_feat_size*4,d_base_feat_size*4), in_out_chan[1], heads[1], n_groups[2], window_size= window_sizes[0], n_class=num_classes)
        #self.decoder_1= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], n_class=num_classes, is_last=True)
        self.decoder_1 = MyDecoderLayer4((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], window_size= window_sizes[0], n_class=num_classes, is_last=True)

    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        bottleneck, skip_inputs, _, _ = self.backbone(x)
        #bridge = self.bridge(encoder) #list

        #b,c,_,_ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        #---------------Decoder-------------------------
        # print("stage3-----")
        #tmp_3 = self.decoder_3(bridge[3].permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")
        #tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0,2,3,1))
        # print("stage1-----")
        #tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0,2,3,1))
        # print("stage0-----")
        #tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0,2,3,1))

        #modified decoder
        x = self.decoder_4(x1 = bottleneck, x2= None)

        x = self.decoder_3(x1 = x ,x2= skip_inputs[2])

        #x = dec3(x , inp1_1)

        x = self.decoder_2(x1 = x ,x2= skip_inputs[1])

        #x = dec4(x , inp0_1)
        x = self.decoder_1(x1 = x ,x2= skip_inputs[0])
        #x = self.decoder_1(x1 = bottleneck ,x2= skip_inputs[0])
        return x
class MISSFormer7(nn.Module):
    def __init__(self, num_classes=9, encoder_pretrained=True):
        super().__init__()

            # reduction_ratios = [8, 4, 2, 1]
            # heads = [1, 2, 5, 8]
        window_sizes = [7, 7, 7, 7]
        heads = [3, 6, 12, 24]
        n_groups = [48, 24, 12, 6]
        d_base_feat_size = 7  # 16 for 512 inputsize   7for 224
        # in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        in_out_chan = [[96, 96], [192, 192], [384, 384], [768, 768]]

            # dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]

        self.backbone = DATMissLG2(img_size=56, patch_size=4, expansion=4, dim_stem=96,
                                    dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                                    heads=[3, 6, 12, 24], window_sizes=[7, 7, 7, 7],
                                    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                                    strides=[1, 1, 1, 1], offset_range_factor=[1, 2, 3, 4],
                                    stage_spec=[['E', "DM"], ['E', "DM"], ['E', "DM", 'E', "DM", 'E', "DM"],['DM', "DM"]],
                                    groups=[1, 1, 3, 6], use_pes=[False, False, False, False],
                                    dwc_pes=[False, False, False, False], sr_ratios=[8, 4, 2, 1],
                                    lower_lr_kvs={}, fixed_pes=[False, False, False, False],
                                    no_offs=[False, False, False, False], ns_per_pts=[4, 4, 4, 4],
                                    use_dwc_mlps=[False, False, False, False], use_conv_patches=True,
                                    ksizes=[9, 7, 5, 3], ksize_qnas=[3, 3, 3, 3], nqs=[2, 2, 2, 2],
                                    qna_activation='exp', nat_ksizes=[3, 3, 3, 3],
                                    layer_scale_values=[-1, -1, -1, -1],
                                    use_lpus=[False, False, False, False], log_cpb=[False, False, False, False])

                # self.reduction_ratios = [1, 2, 4, 8]
                # self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)

                # dec_4 = MyDecoderLayer(input_size = (7,7), in_out_chan= (768,768), heads = 24, n_groups = 48, is_first = True)
                # dec_3 = MyDecoderLayer(input_size = (14,14), in_out_chan= (384,384), heads = 12, n_groups = 24)
                # ec_2 = MyDecoderLayer(input_size = (28,28), in_out_chan= (192,192), heads = 6, n_groups = 12)
                # dec_1 = MyDecoderLayer(input_size = (56,56), in_out_chan= (96,96), heads = 3, n_groups = 6, is_last = True)

        self.decoder_4 = MyDecoderLayer1((d_base_feat_size, d_base_feat_size), in_out_chan[3], heads[3],
                                                 n_groups[0], n_class=num_classes, is_first=True)
                # self.decoder_3= MyDecoderLayer1((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], n_groups[1], window_size= window_sizes[0], n_class=num_classes)
        self.decoder_3 = MyDecoderLayer2Deep((d_base_feat_size * 2, d_base_feat_size * 2), in_out_chan[2],
                                                     heads[2], n_groups[1], window_size=window_sizes[0],
                                                     n_class=num_classes)
        self.decoder_2 = MyDecoderLayer4((d_base_feat_size * 4, d_base_feat_size * 4), in_out_chan[1], heads[1],
                                                 n_groups[2], window_size=window_sizes[0], n_class=num_classes)
                # self.decoder_1= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], n_groups[3], n_class=num_classes, is_last=True)
        self.decoder_1 = MyDecoderLayer4((d_base_feat_size * 8, d_base_feat_size * 8), in_out_chan[0], heads[0],
                                                 n_groups[3], window_size=window_sizes[0], n_class=num_classes,
                                                 is_last=True)

        def forward(self, x):
                # ---------------Encoder-------------------------
            if x.size()[1] == 1:
                x = x.repeat(1, 3, 1, 1)

                bottleneck, skip_inputs, _, _ = self.backbone(x)
                # bridge = self.bridge(encoder) #list

                # b,c,_,_ = bridge[3].shape
                # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
                # ---------------Decoder-------------------------
                # print("stage3-----")
                # tmp_3 = self.decoder_3(bridge[3].permute(0,2,3,1).view(b,-1,c))
                # print("stage2-----")
                # tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0,2,3,1))
                # print("stage1-----")
                # tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0,2,3,1))
                # print("stage0-----")
                # tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0,2,3,1))

                # modified decoder
                x = self.decoder_4(x1=bottleneck, x2=None)

                x = self.decoder_3(x1=x, x2=skip_inputs[2])

                # x = dec3(x , inp1_1)

                x = self.decoder_2(x1=x, x2=skip_inputs[1])

                # x = dec4(x , inp0_1)
                x = self.decoder_1(x1=x, x2=skip_inputs[0])
                # x = self.decoder_1(x1 = bottleneck ,x2= skip_inputs[0])

            return x
if __name__ == "__main__":
    missu = MISSFormer6()

    akharin = torch.randn(1, 3, 224, 224)

    print(missu(akharin).shape)