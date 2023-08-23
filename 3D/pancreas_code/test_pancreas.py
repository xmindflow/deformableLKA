import os
import argparse
import torch
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from networks.unetr import UNETR
from networks.d_lka_former.d_lka_net_synapse import D_LKA_Net
from networks.d_lka_former.transformerblock import TransformerBlock_3D_single_deform_LKA, TransformerBlock

from test_util import test_all_case

from fvcore.nn import FlopCountAnalysis
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/work/scratch/niggemeier/projects/MCF/dataset_pancreas/', help='Name of Experiment')  # todo change dataset path
parser.add_argument('--model', type=str,  default="pancreas1", help='model_name')                # todo change test model name
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = "./model/"+args.model+'/'
test_save_path = "./model/prediction/unetrpp/"+args.model+"_post/" # change test save directory here
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(args.root_path + '/Pancreas/Flods/test0.list', 'r') as f:                                         # todo change test flod
    image_list = f.readlines()
image_list = [args.root_path +item.replace('\n', '') for item in image_list] #+"/mri_norm2.h5"

def create_model(name='dlka_former'):
    # Network definition        
    if name == 'dlka_former':
            
            net = D_LKA_Net(in_channels=1, 
                           out_channels=num_classes, 
                           img_size=[96, 96, 96],
                           patch_size=(2,2,2),
                           input_size=[48*48*48, 24*24*24,12*12*12,6*6*6],
                           trans_block=TransformerBlock,
                           do_ds=False)
            model = net.cuda()

    return model

def test_calculate_metric(epoch_num):
    dlka_former   = create_model(name='dlka_former')

    lka_save_mode_path = os.path.join(snapshot_path, 'd_lka_former_iter_' + str(epoch_num) + '.pth')
    dlka_former.load_state_dict(torch.load(lka_save_mode_path))
    print("init weight from {}".format(lka_save_mode_path))
    
    n_parameters = sum(p.numel() for p in dlka_former.parameters() if p.requires_grad)
    input_res = (1, 96, 96, 96)
    input = torch.ones(()).new_empty((1, *input_res), dtype=next(dlka_former.parameters()).dtype,
                                        device=next(dlka_former.parameters()).device)
    flops = FlopCountAnalysis(dlka_former, input)
    model_flops = flops.total()
    print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
    print(f"MAdds: {round(model_flops * 1e-9, 2)} G")

    dlka_former.eval()


    avg_metric = test_all_case(dlka_former, dlka_former, image_list, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    iters = 6000
    metric = test_calculate_metric(iters)
    print('iter:', iters)
    print(metric)
