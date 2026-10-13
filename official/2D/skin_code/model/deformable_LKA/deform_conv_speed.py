import torch
from torch import nn
import torchvision
from mmcv.ops import DeformConv2d
#from ops.dcn import DeformConv
import time


class NormalConv(nn.Module):

    def __init__(self, in_channels, groups):
        super(NormalConv, self).__init__()
        kernel_size = (3, 3)
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    bias=True)

        self.deform_conv = nn.Conv2d(in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=kernel_size,
                                     padding=1,
                                     groups=groups,
                                     bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x)
        return out


class DeformConvTorchvision(nn.Module):

    def __init__(self, in_channels, groups):
        super(DeformConvTorchvision, self).__init__()
        kernel_size = (3, 3)
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=1,
                                                        groups=groups,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

"""
class DeformConvMmdet(nn.Module):

    def __init__(self, in_channels, groups):
        super(DeformConvMmdet, self).__init__()
        kernel_size = (3, 3)

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    bias=True)

        self.deform_conv = DeformConv(in_channels=in_channels,
                                      out_channels=in_channels,
                                      kernel_size=kernel_size,
                                      padding=1,
                                      groups=groups,
                                      bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out
"""        

from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np


class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformConv3d, self).__init__()
        self.kernel_size = kernel_size
        N = kernel_size ** 3
        self.stride = stride
        self.padding = padding
        self.zero_padding = nn.ConstantPad3d(padding, 0)
        self.conv_kernel = nn.Conv3d(in_channels * N, out_channels, kernel_size=1, bias=bias)
        self.offset_conv_kernel = nn.Conv3d(in_channels, N * 3, kernel_size=kernel_size, padding=padding, bias=bias)
        
        self.mode = "deformable"
        
    def deformable_mode(self, on=True): # 
        if on:
            self.mode = "deformable"
        else:
            self.mode = "regular"
        
    def forward(self, x):
        if self.mode == "deformable":
            offset = self.offset_conv_kernel(x)
        else:
            b, c, h, w, d = x.size()
            offset = torch.zeros(b, 3 * self.kernel_size ** 3, h, w, d).to(x)
        
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 3

        if self.padding:
            x = self.zero_padding(x)

        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)
        p = p[:, :, ::self.stride, ::self.stride, ::self.stride]

        # (b, h, w, d, 3N), N == ks ** 3, 3N - 3 coords for each point on the activation map
        p = p.contiguous().permute(0, 2, 3, 4, 1) # 5D array
        
        q_sss = Variable(p.data, requires_grad=False).floor() # point with all smaller coords
#         q_sss = p.data.floor() - same? / torch.Tensor(p.data).floor()
        q_lll = q_sss + 1 # all larger coords

        # 8 neighbor points with integer coords
        q_sss = torch.cat([
            torch.clamp(q_sss[..., :N], 0, x.size(2) - 1), # h_coord
            torch.clamp(q_sss[..., N:2 * N], 0, x.size(3) - 1), # w_coord
            torch.clamp(q_sss[..., 2 * N:], 0, x.size(4) - 1) # d_coord
        ], dim=-1).long()
        q_lll = torch.cat([
            torch.clamp(q_lll[..., :N], 0, x.size(2) - 1), # h_coord
            torch.clamp(q_lll[..., N:2 * N], 0, x.size(3) - 1), # w_coord
            torch.clamp(q_lll[..., 2 * N:], 0, x.size(4) - 1) # d_coord
        ], dim=-1).long()
        q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)

        # (b, h, w, d, N)
        mask = torch.cat([
            p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
            p[..., N:2 * N].lt(self.padding) + p[..., N:2 * N].gt(x.size(3) - 1 - self.padding),
            p[..., 2 * N:].lt(self.padding) + p[..., 2 * N:].gt(x.size(4) - 1 - self.padding),
        ], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p)) # все еще непонятно, что тут происходит за wtf
        p = p * (1 - mask) + floor_p * mask
        
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),
            torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1),
        ], dim=-1)
        
        # trilinear kernel (b, h, w, d, N)  
        g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        
        # get values in all 8 neighbor points
        # (b, c, h, w, d, N) - 6D-array
        x_q_sss = self._get_x_q(x, q_sss, N)
        x_q_lll = self._get_x_q(x, q_lll, N)
        x_q_ssl = self._get_x_q(x, q_ssl, N)
        x_q_sls = self._get_x_q(x, q_sls, N)
        x_q_sll = self._get_x_q(x, q_sll, N)
        x_q_lss = self._get_x_q(x, q_lss, N)
        x_q_lsl = self._get_x_q(x, q_lsl, N)
        x_q_lls = self._get_x_q(x, q_lls, N)
        
        # (b, c, h, w, d, N)
        x_offset = g_sss.unsqueeze(dim=1) * x_q_sss + \
                   g_lll.unsqueeze(dim=1) * x_q_lll + \
                   g_ssl.unsqueeze(dim=1) * x_q_ssl + \
                   g_sls.unsqueeze(dim=1) * x_q_sls + \
                   g_sll.unsqueeze(dim=1) * x_q_sll + \
                   g_lss.unsqueeze(dim=1) * x_q_lss + \
                   g_lsl.unsqueeze(dim=1) * x_q_lsl + \
                   g_lls.unsqueeze(dim=1) * x_q_lls
        
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)
        
        return out
    
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), 
            indexing='ij')
        
        # (3N, 1) - 3 coords for each of N offsets
        # (x1, ... xN, y1, ... yN, z1, ... zN)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)
        
        return p_n
    
    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)

        return p_0
    
    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype).to(offset)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype).to(offset)
        p = p_0 + p_n + offset
        
        return p
    
    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()
        
        #           (0, 1, 2, 3, 4)
        # x.size == (b, c, h, w, d)
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
        # (b, c, h*w*d)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        # offset_x * w * d + offset_y * d + offset_z
        index = q[..., :N] * padded_w * padded_d + q[..., N:2 * N] * padded_d + q[..., 2 * N:]
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)
        
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)
        
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, d, N = x_offset.size()
        x_offset = x_offset.permute(0, 1, 5, 2, 3, 4)
        x_offset = x_offset.contiguous().view(b, c * N, h, w, d)

        return x_offset


def measure_time(net, input, n_times):
    net.eval()
    warm_up = 20
    sum_time = 0
    for i in range(warm_up + n_times):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = net(input)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warm_up:
            sum_time += (t1 - t0)

    return (sum_time * 1000 / n_times)

def measure_time_backward_simple(net, input, loss_func, n_times, sigmoid):
    net.train()
    warm_up = 20
    sum_time = 0
    for i in range(warm_up + n_times):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = net(input)
        out_sigmoid = sigmoid(out)
        input_sigmoid = sigmoid(input)
        loss = loss_func(out_sigmoid, input_sigmoid)
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warm_up:
            sum_time += (t1 - t0)

    return (sum_time * 1000 / n_times)

from deformable_3dconvolution import deform_conv3d

def test(bs, groups, in_channels, n_times=100):
    device = torch.device('cuda')
    w, h, d = 32, 32, 32
    input = torch.rand(bs, in_channels, h, w).to(device)
    input3d = torch.rand((1,32,32,32,32)).to(device) # b c h w d

    normal_conv = NormalConv(in_channels, groups).to(device)
    def_conv_torchvision = DeformConvTorchvision(in_channels, groups).to(device)
    #def_conv_mmdet = DeformConvMmdet(in_channels, groups).to(device)
    deform_conv3d_1 = DeformConv3d(in_channels=32, out_channels=32, kernel_size=3).to(device)

    deform_conv3d_2 = deform_conv3d(in_c=32, out_c=32, kernel_size=3).to(device)

    time_normal_conv = measure_time(normal_conv, input, n_times)
    time_torchvision = measure_time(def_conv_torchvision, input, n_times)
    time_conv3d = measure_time(deform_conv3d_1, input3d, n_times)
    time_conv3d_2 = measure_time(deform_conv3d_2, input3d, n_times)
    #time_mmdet = measure_time(def_conv_mmdet, input, n_times)
    
    #print(f"{'Time normal conv:':<30} {time_normal_conv:>6.2f} ms")
    #print(f"{'Time torchvision deform conv:':<30} {time_torchvision:>6.2f} ms")
    print(f"{'Time deform conv 3d 1:':<30} {time_conv3d:>6.2f} ms")
    print(f"{'Time deform conv 3d 2:':<30} {time_conv3d_2:>6.2f} ms")
    #print(f"{'Time mmdet deform conv:':<30} {time_mmdet:>6.2f} ms")

def test_backward_simple(bs, groups, in_channels, n_times=1000):
    device = torch.device('cuda')

    conv_3d = nn.Conv3d(32,32,3,1,1).to(device)
    deform_conv_pack = DeformConv3d(32,32,3,1,1).to(device)
 
    input3d = torch.rand((1,32,32,32,32), requires_grad=True).to(device)

    lossfunc = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    time_deform_conv = measure_time_backward_simple(deform_conv_pack, input3d, lossfunc, n_times, sigmoid)
    time_conv_3d = measure_time_backward_simple(conv_3d, input3d, lossfunc, n_times, sigmoid)

    print(f"{'Time backward deform conv 3d:':<30} {time_deform_conv:>6.2f} ms")
    print(f"{'Time torch conv 3d:':<30} {time_conv_3d:>6.2f} ms")

if __name__ == "__main__":
    in_channels = 512
    bs_list = [1, 1, 16, 16]
    groups_list = [1, in_channels, 1, in_channels]

    with torch.no_grad():
        for bs, groups in zip(bs_list, groups_list):
            #print(f"bs: {bs:02d}, in-channels: {in_channels}, groups: {groups}")
            test(bs, groups, in_channels, n_times=100)
            print("----------------------------------------")

    for bs, groups in zip(bs_list, groups_list):
        #print(f"bs: {bs:02d}, in-channels: {in_channels}, groups: {groups}")
        test_backward_simple(bs, groups, in_channels, n_times=100)
        print("----------------------------------------")