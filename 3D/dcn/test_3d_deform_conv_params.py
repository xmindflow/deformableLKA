import torch
from torch import nn
import torchvision
#from ops.dcn import DeformConv
import time

import torch.optim

from modules.deform_conv import DeformConv, _DeformConv, DeformConvPack, DeformConvPack_experimental
from modules.deform_conv import DeformConv_d, _DeformConv, DeformConvPack_d

from fvcore.nn import FlopCountAnalysis

deform_Conv_3D = DeformConvPack(64,64,kernel_size=(5,5,5), padding=6, dilation=3, stride=1).cuda(0)
deform_Conv_3D_exp = DeformConvPack_experimental(64,64,kernel_size=(7,7,7), padding=9, dilation=3, stride=1).cuda(0)

input = torch.rand((1,64,16,16,16)).cuda(0)

flops = FlopCountAnalysis(deform_Conv_3D, input)
flops_exp = FlopCountAnalysis(deform_Conv_3D_exp, input)

n_parameters = sum(p.numel() for p in deform_Conv_3D.parameters() if p.requires_grad)
n_parameters_exp = sum(p.numel() for p in deform_Conv_3D_exp.parameters() if p.requires_grad)

model_flops = flops.total()
model_flops_exp = flops_exp.total()

print(f"Total trainable parameters: {round(n_parameters * 1e-3, 4)} K")
print(f"MAdds: {round(model_flops * 1e-6, 4)} M")
print(f"Total trainable parameters exp: {round(n_parameters_exp * 1e-3, 4)} K")
print(f"MAdds exp: {round(model_flops_exp * 1e-6, 4)} M")