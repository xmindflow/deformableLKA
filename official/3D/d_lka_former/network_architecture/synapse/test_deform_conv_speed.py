import torch
from torch import nn
import torchvision
#from ops.dcn import DeformConv
import time


from deform_conv import DeformConvPack

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

def measure_time(net, input, n_times):
    net.train()
    warm_up = 20
    sum_time = 0
    for i in range(warm_up + n_times):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = net(input)
        
        out.backward(input)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warm_up:
            sum_time += (t1 - t0)

    return (sum_time * 1000 / n_times)


def test(bs, groups, in_channels, n_times=1000):
    device = torch.device('cuda')
    w, h = 13, 13
    input = torch.rand(bs, in_channels, h, w).to(device)

    normal_conv = NormalConv(in_channels, groups).to(device)
    def_conv_torchvision = DeformConvTorchvision(in_channels, groups).to(device)
    #def_conv_mmdet = DeformConvMmdet(in_channels, groups).to(device)
    deform_conv_pack = DeformConvPack(32,32,3,1,1).to(device)
    input3d = torch.rand((1,32,32,32,32), requires_grad=True).to(device)
    conv_3d = nn.Conv3d(32,32,3,1,1).to(device)

    #time_normal_conv = measure_time(normal_conv, input, n_times)
    #time_torchvision = measure_time(def_conv_torchvision, input, n_times)
    #time_mmdet = measure_time(def_conv_mmdet, input, n_times)
    time_deform_conv = measure_time(deform_conv_pack, input3d, n_times)
    time_conv_3d = measure_time(conv_3d, input3d, n_times)
    #print(f"{'Time normal conv:':<30} {time_normal_conv:>6.2f} ms")
    #print(f"{'Time torchvision deform conv:':<30} {time_torchvision:>6.2f} ms")
    print(f"{'Time  deform conv 3d:':<30} {time_deform_conv:>6.2f} ms")
    print(f"{'Time torch conv 3d:':<30} {time_conv_3d:>6.2f} ms")
    #print(f"{'Time mmdet deform conv:':<30} {time_mmdet:>6.2f} ms")


if __name__ == "__main__":
    in_channels = 512
    bs_list = [1, 1, 16, 16]
    groups_list = [1, in_channels, 1, in_channels]

    with torch.no_grad():
        for bs, groups in zip(bs_list, groups_list):
            #print(f"bs: {bs:02d}, in-channels: {in_channels}, groups: {groups}")
            test(bs, groups, in_channels, n_times=100)
            print("----------------------------------------")