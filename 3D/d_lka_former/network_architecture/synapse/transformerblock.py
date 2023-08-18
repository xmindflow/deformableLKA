import torch.nn as nn
import torch
from d_lka_former.network_architecture.dynunet_block import UnetResBlock


class TransformerBlock(nn.Module): # Rename
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class EPA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)

        x = torch.cat((x_SA, x_CA), dim=-1)
        
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
    


from torch.nn import functional as F
import sys
class EfficientAttention(nn.Module):
    """
    input  -> x:[B, N, C]
    output ->   [B, N, C]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, input_size, hidden_size, head_count=4, qkv_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_count = head_count

        self.key_lin = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.query_lin = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.value_lin = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.reprojection = nn.Linear(hidden_size, hidden_size)

        self.temperature = nn.Parameter(torch.ones(head_count, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(head_count, 1, 1))

    def forward(self, input_):
        B, N, C = input_.shape
        #print("Input shape {}".format(input_.shape))

        queries = self.query_lin(input_).permute(0, 2, 1)
        #print("queries shape {}".format(queries.shape))
        keys = self.key_lin(input_).permute(0, 2, 1)
        #print("keys shape {}".format(keys.shape))
        values = self.value_lin(input_).permute(0, 2, 1)
        #print("values shape {}".format(values.shape))

        head_key_channels = self.hidden_size // self.head_count
        head_value_channels = self.hidden_size // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            #print("Key shape: {}".format(key.shape))

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            #print("Query shape: {}".format(query.shape))
            #print("Query transposed shape: {}".format(query.transpose(1,2).shape))

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            #print("Value shape: {}".format(value.shape))
            #print("Value transposed shape: {}".format(value.transpose(1,2).shape))

            context = key @ value.transpose(1, 2)  # dk*dv
            #print("Context shape: {}".format(context.shape))
            #print("Context transposed shape: {}".format(context.transpose(1,2).shape))
            attended_value = (context.transpose(1, 2) @ query) # n*dv
            #print("Attended value shape: {}".format(attended_value.shape))
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        #print("Aggregated values shape before reprojection: {}".format(aggregated_values.shape))
        attention = self.reprojection(aggregated_values.transpose(1,2))
        #print("Aggregated values shape after reprojection: {}".format(attention.shape))
        #sys.exit()
        return attention
    

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
    


class TransformerBlock_EA(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EfficientAttention(input_size=input_size, hidden_size=hidden_size, head_count=num_heads)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
    

#########################
#
# 3D LKA
# 
#########################
class TransformerBlock_3D_LKA(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using LKA Attention with different Kernel sizes.")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = LKA_Attention3d(d_model=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H , W , D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        kernel_dwd = 7
        dilation_dwd = 3
        padding_dwd = 9
        kernel_dw = 5
        padding_dw = 2
        '''
        if dim == 32 or dim == 64:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim == 128:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim == 256:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))
        '''
        

        self.conv0 = nn.Conv3d(dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=kernel_dwd, stride=1, padding=padding_dwd, groups=dim, dilation=dilation_dwd)
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x


###########################
#
# Transformer Block 2D deformable convolution
#
###########################
import torchvision

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5,5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7,7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shorcut = x.clone()
        x_copy = x.clone()

        # Shape B C H W D
        # Extract Depths
        for i in range(x.size(-1)):
            x_temp = x[:,:,:,:,i]
            #print(x_temp.shape)
            x_temp = self.proj_1(x_temp)
            x_temp = self.activation(x_temp)
            x_temp = self.spatial_gating_unit(x_temp)
            x_temp = self.proj_2(x_temp)
            x_copy[:,:,:,:,i] = x_temp

        #print("X shape after loop:{}".format(x.shape))
        #print("Shorcut shape after loop:{}".format(shorcut.shape))
        x = x_copy + shorcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1) # B N C
        return x

class TransformerBlock_2Dsingle(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        #print("Using LKA Attention")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = deformable_LKA_Attention(d_model=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        y = self.norm(x)
        #print(y.shape)
        z = self.epa_block(y, B, C, H, W, D)
        attn = x + self.gamma * z
        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


#########################
#
# 3D LKA with one deform conv
# 
#########################
from d_lka_former.network_architecture.synapse.deform_conv import DeformConvPack, DeformConvPack_Depth

class TransformerBlock_3D_single_deform_LKA(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using LKA Attention with one deformable layer")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = LKA_Attention3d_deform(d_model=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H , W , D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_deform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.deform_conv = DeformConvPack(in_channels=dim, out_channels=dim, kernel_size=(3,3,3), stride=1, padding=1)
        #print("Using single deformable layers.")
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_deform(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_deform(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x


#########################
#
# 3D LKA with additional 3D conv
# 
#########################

class TransformerBlock_3D_LKA_3D_conv(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using LKA Attention with additional conv layer")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = LKA_Attention3d_conv(d_model=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H , W , D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_conv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.deform_conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(3,3,3), stride=1, padding=1)
        #print("Using single deformable layers.")
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_conv(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_conv(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x


###################################################
#
# 3D LKA and spatial attention
# 
###################################################

class SpatialAttention_LKA(nn.Module):
    """
        Spatial attention parallel to 3d LKA
    """
    def __init__(self, input_size, hidden_size, num_heads=4, qkv_bias=False, proj_size=32,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.lka = LKA_Attention3d_withSpatial(d_model=hidden_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape

        # Channel attention

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, v_SA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_projected = self.E(key)

        v_SA_projected = self.F(v_SA)

        query = torch.nn.functional.normalize(query, dim=-1)
        #key = torch.nn.functional.normalize(key, dim=-1)

        attn_SA = (query.permute(0, 1, 3, 2) @ k_projected) * self.temperature

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # LKA
        x_LKA = self.lka(x, B, C, H, W, D)


        # Concat fusion
        x_LKA = self.out_proj(x_LKA)
        x_SA = self.out_proj2(x_SA)

        #print("x_LKA shape: {}".format(x_LKA.shape))
        #print("x_SA shape: {}".format(x_SA.shape))
        
        x = torch.cat((x_SA, x_LKA), dim=-1) 
        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class TransformerBlock_LKA_Spatial(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and channel attention
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using LKA and Spatial Attention")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = SpatialAttention_LKA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_withSpatial(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_Spatial(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x



###################################################
#
# 3D LKA and channel attention
# 
###################################################

class ChannelAttention_LKA(nn.Module):
    """
        Channel attention parallel to 3d LKA
    """
    def __init__(self, hidden_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.lka = LKA_Attention3d_withChannel(d_model=hidden_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape

        # Channel attention

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, v_CA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attn_CA = (query @ key.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        # LKA
        x_SA = self.lka(x, B, C, H, W, D)


        # Concat fusion
        x_CA = self.out_proj(x_CA)
        x_SA = self.out_proj2(x_SA)
        
        x = torch.cat((x_SA, x_CA), dim=-1) 
        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class TransformerBlock_LKA_Channel(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and channel attention
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using LKA and Channel Attention")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = ChannelAttention_LKA(hidden_size=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_withChannel(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_Channel(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x

###################################################
#
# 3D LKA and channel attention with more layer norms
# 
###################################################

class ChannelAttention_LKA_norm(nn.Module):
    """
        Channel attention parallel to 3d LKA
    """
    def __init__(self, hidden_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(1, 1, 1))

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.lka = LKA_Attention3d_withChannel_norm(d_model=hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape

        # Channel attention

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, v_CA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attn_CA = (query @ key.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        # LKA
        x_SA = self.lka(x, B, C, H, W, D) * self.temperature2


        # Concat fusion
        x_CA = self.out_proj(self.norm(x_CA))
        x_SA = self.out_proj2(self.norm2(x_SA))
        
        x = torch.cat((x_SA, x_CA), dim=-1) 
        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class TransformerBlock_LKA_Channel_norm(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and channel attention
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using LKA and Channel Attention")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = ChannelAttention_LKA_norm(hidden_size=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_Channel_norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_withChannel_norm(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_Channel_norm(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x
    

#########################
#
# 3D LKA with SE Module
# 
#########################
class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 4, rd_channels=None, bias=True):
        super(SEModule, self).__init__()
        if rd_channels is None:
            rd_channels = int(channels * rd_ratio)
        self.fc1 = nn.Conv3d(channels, rd_channels, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()
        print("Using SE Module")

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True) # B C H W D --> B C 1 1 1
        x_se = self.fc1(x_se) # B C 1 1 1
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class TransformerBlock_SE(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        #print("Using LKA Attention")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.se = SEModule(channels=hidden_size, rd_ratio=1./4)
        self.LKA_block = LKA_Attention3d_SE(d_model=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D
        x = self.se(x)
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        attn = x + self.gamma * self.LKA_block(self.norm(x), B, C, H , W , D)
        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_SE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_SE(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_SE(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x
    

###################################################
#
# 3D deform LKA and channel attention
# 
###################################################

class ChannelAttention_Deform_LKA(nn.Module):
    """
        Channel attention parallel to 3d LKA
    """
    def __init__(self, hidden_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.lka = LKA_Attention3d_Deform_withChannel(d_model=hidden_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape

        # Channel attention

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, v_CA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attn_CA = (query @ key.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        # LKA
        x_SA = self.lka(x, B, C, H, W, D)


        # Concat fusion
        x_CA = self.out_proj(x_CA) # We trained it with only one linear layer....
        x_SA = self.out_proj2(x_SA)
        
        x = torch.cat((x_SA, x_CA), dim=-1) 
        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class TransformerBlock_Deform_LKA_Channel(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and channel attention
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using Deform LKA and Channel Attention")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = ChannelAttention_Deform_LKA(hidden_size=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_Deform_Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()

        kernel_dwd = 7
        dilation_dwd = 3
        padding_dwd = 9
        kernel_dw = 5
        padding_dw = 2
        """
        if dim == 32 or dim == 64:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim == 128:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim == 256:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))
        """

        self.conv0 = nn.Conv3d(dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=kernel_dwd, stride=1, padding=padding_dwd, groups=dim, dilation=dilation_dwd)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        
        self.deform_conv = DeformConvPack(in_channels=dim, out_channels=dim, kernel_size=(3,3,3), stride=1, padding=1)
        

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_Deform_withChannel(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_Deform_Channel(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x
    

###################################################
#
# 3D deform LKA and channel attention Sequential
# 
###################################################

class ChannelAttention_Deform_LKA_sequential(nn.Module):
    """
        Channel attention parallel to 3d LKA
    """
    def __init__(self, hidden_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.norm = nn.LayerNorm(hidden_size) # operates on 'b n c' eith c = hidden_size
        self.lka = LKA_Attention3d_Deform_withChannel_sequential(d_model=hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape

        # Channel attention

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, v_CA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attn_CA = (query @ key.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C) # B N C
        x_CA = self.norm(x_CA)
        x_CA = x_CA.permute(0, 2, 1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D

        # LKA
        x_SA = self.lka(x_CA, B, C, H, W, D)

        x_SA = x_SA.reshape(B, C, H*W*D).permute(0, 2, 1) # B C H W D --> B C N --> B N C 

        x_SA = self.norm2(x_SA)

        #x_SA = x_SA.permute(0, 2, 1) # B N C --> B C N

        # Concat fusion
        x = self.out_proj(x_SA)

        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class TransformerBlock_Deform_LKA_Channel_sequential(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and channel attention
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using Deform LKA and Channel Attention SEQUENTIAL")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = ChannelAttention_Deform_LKA_sequential(hidden_size=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_Deform_Channel_sequential(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 32 or dim == 64:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim == 128:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim == 256:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))

        self.conv0 = nn.Conv3d(dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=kernel_dwd, stride=1, padding=padding_dwd, groups=dim, dilation=dilation_dwd)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        
        self.deform_conv = DeformConvPack(in_channels=dim, out_channels=dim, kernel_size=(3,3,3), stride=1, padding=1)
        

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_Deform_withChannel_sequential(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_Deform_Channel_sequential(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        #x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x


###################################################
#
# 3D deform LKA and Spatial attention Sequential
# 
###################################################

class SpatialAttention_Deform_LKA_sequential(nn.Module):
    """
        Spatial attention sequential to 3d LKA
    """
    def __init__(self, input_size, hidden_size, proj_size=32, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)
        
        self.norm = nn.LayerNorm(hidden_size) # operates on 'b n c' eith c = hidden_size
        self.lka = LKA_Attention3d_Deform_withSpatial_sequential(d_model=hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape

        # Channel attention

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, v_SA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_projected = self.E(key)

        v_SA_projected = self.F(v_SA)

        query = torch.nn.functional.normalize(query, dim=-1)
        #key = torch.nn.functional.normalize(key, dim=-1)

        attn_SA = (query.permute(0, 1, 3, 2) @ k_projected) * self.temperature

        attn_SA = attn_SA.softmax(dim=-1)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        x_SA = self.norm(x_SA)
        x_SA = x_SA.permute(0, 2, 1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D

        # LKA
        x_LKA = self.lka(x_SA, B, C, H, W, D)

        x_LKA = x_LKA.reshape(B, C, H*W*D).permute(0, 2, 1) # B C H W D --> B C N --> B N C 

        x_LKA = self.norm2(x_LKA)

        #x_SA = x_SA.permute(0, 2, 1) # B N C --> B C N

        # Concat fusion
        x = self.out_proj(x_LKA)

        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class TransformerBlock_Deform_LKA_Spatial_sequential(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and spatial attention
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using Deform LKA and Spatial Attention SEQUENTIAL")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = SpatialAttention_Deform_LKA_sequential(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_Deform_Spatial_sequential(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 32 or dim == 64:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim == 128:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim == 256:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))

        self.conv0 = nn.Conv3d(dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=kernel_dwd, stride=1, padding=padding_dwd, groups=dim, dilation=dilation_dwd)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        
        self.deform_conv = DeformConvPack(in_channels=dim, out_channels=dim, kernel_size=(3,3,3), stride=1, padding=1)
        

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_Deform_withSpatial_sequential(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_Deform_Spatial_sequential(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        #x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x
    
    
    
###################################################
#
# 3D deform LKA and spatial attention
# 
###################################################

class SpatialAttention_Deform_LKA(nn.Module):
    """
        Spatial attention parallel to 3d LKA
    """
    def __init__(self, input_size, hidden_size, num_heads=4, qkv_bias=False, proj_size=32,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.lka = LKA_Attention3d_Deform_withSpatial(d_model=hidden_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape

        # Channel attention

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, v_SA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_projected = self.E(key)

        v_SA_projected = self.F(v_SA)

        query = torch.nn.functional.normalize(query, dim=-1)
        #key = torch.nn.functional.normalize(key, dim=-1)

        attn_SA = (query.permute(0, 1, 3, 2) @ k_projected) * self.temperature

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # LKA
        x_LKA = self.lka(x, B, C, H, W, D)


        # Concat fusion
        x_LKA = self.out_proj(x_LKA) # We trained it with only one linear layer....
        x_SA = self.out_proj2(x_SA)
        
        x = torch.cat((x_SA, x_LKA), dim=-1) 
        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class TransformerBlock_Deform_LKA_Spatial(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and spatial attention
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using Deform LKA kernel sizes and Spatial Attention")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = SpatialAttention_Deform_LKA(input_size=input_size,hidden_size=hidden_size, proj_size=proj_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x



class LKA3d_Deform_Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 32 or dim == 64:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim == 128:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim == 256:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))

        self.conv0 = nn.Conv3d(dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=kernel_dwd, stride=1, padding=padding_dwd, groups=dim, dilation=dilation_dwd)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        
        self.deform_conv = DeformConvPack(in_channels=dim, out_channels=dim, kernel_size=(3,3,3), stride=1, padding=1)
        

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d_Deform_withSpatial(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d_Deform_Spatial(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x