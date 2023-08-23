import torch
import numpy as np
from networks.MaxViT_LKA_Decoder import MaxViTLKAFormer
from networks.MaxViT_deform_LKA import MaxViT_deformableLKAFormer
from fvcore.nn import FlopCountAnalysis
import time
"""
From : https://deci.ai/blog/measure-inference-time-deep-neural-networks/
"""
net = MaxViTLKAFormer().cuda()
#net = MaxViT_deformableLKAFormer().cuda()
net.eval()
input = torch.rand((1,3,224,224)).cuda()

n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
flops = FlopCountAnalysis(net, input)
model_flops = flops.total()
print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
print(f"MAdds: {round(model_flops * 1e-9, 2)} G")


# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 1000
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(50):
    _ = net(input)
print("Warmup done, now measuring performance.")
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        #print(rep)
        starter.record()
        _ = net(input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(f"Mean time: {mean_syn}")
print(f"Std deviation: {std_syn}")

# second method

with torch.no_grad():
    start_time = time.time()
    for rep in range(repetitions):
        _ = net(input)

    end_time = time.time()

print(f"Second time measurement: {(end_time-start_time)/repetitions}")