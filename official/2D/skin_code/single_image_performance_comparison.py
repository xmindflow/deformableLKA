# This is a script to compare different methods and their dice scores on single images.
# By that we can find images, where our methods is superior in comparison to other methods.
from __future__ import division
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loader import *
import pandas as pd
import glob
import argparse
import nibabel as nib
import numpy as np
from sklearn.metrics import f1_score
import copy
import yaml
import torch
from model.MaxViT_LKA_Decoder import MaxViTLKAFormer
from model.MaxViT_deform_LKA import MaxViT_deformableLKAFormer
from model.hiformer.HiFormer import HiFormer
from model.hiformer.HiFormer_configs import get_hiformer_b_configs
from model.swinunet.vision_transformer import SwinUnet
from model.swinunet.config import get_config

from tqdm import tqdm
from scipy.ndimage import binary_fill_holes, binary_opening

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./ISIC2018/', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='isic2018', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--cfg', type=str, default='/work/scratch/niggemeier/projects/transnorm/model/swinunet/configs/swin_tiny_patch4_window7_224_lite.yaml', required=False, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()

## Loader
## Hyper parameters
config         = yaml.load(open('./config_skin.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: {}".format(device))
data_path = args.root_path

test_dataset = isic_loader(path_Data = data_path, train = False, Test = True)
test_loader  = DataLoader(test_dataset, batch_size = 1, shuffle= False)
print("Created loaders.")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
dataset_name = args.dataset
dataset_config = {
    'Synapse': {
        'root_path': '',
        'list_dir': '',
        'num_classes': 1,
    },
}
args.num_classes = 1
args.root_path = ''
args.list_dir = ''
args.is_pretrain = True
args.exp = 'Allnets' + dataset_name + str(args.img_size)
snapshot_path = "./mixed_results/{}/{}".format(args.exp, 'model')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
snapshot_path = snapshot_path + '_imgsize'+str(args.img_size)
snapshot_path = snapshot_path + '_s'+str(args.seed)

print("Snapshot path: "+ snapshot_path)

print("Batch size: {}".format(args.batch_size))

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
    
# Networks
net1 = MaxViT_deformableLKAFormer(num_classes=1, img_size=args.img_size)
net2 = MaxViTLKAFormer(num_classes=1, img_size=args.img_size)
net3 = HiFormer(config=get_hiformer_b_configs(),n_classes=1, img_size=args.img_size)
swin_config = get_config(args)
net4 = SwinUnet(config=swin_config, img_size=args.img_size, num_classes=1)
net4.load_from(swin_config)
print("Created networks")

# Load checkpoint
ckpt_1 = torch.load('/work/scratch/niggemeier/projects/transnorm/model_results2018/MaxViT_isic2018224/model_pretrain_epo100_bs16_lr0.01_imgsize224_s200/best_model.pth')
ckpt_2 = torch.load('/work/scratch/niggemeier/projects/transnorm/model_results2018_LKA/MaxViT_isic2018224/model_pretrain_epo100_bs16_lr0.05_imgsize224_s1234/best_model.pth')
ckpt_3 = torch.load('/work/scratch/niggemeier/projects/transnorm/model_results2018/Hiformerisic2018224/model_pretrain_epo100_bs16_lr0.01_imgsize224_s1234/best_model.pth')
ckpt_4 = torch.load('/work/scratch/niggemeier/projects/transnorm/model_results2018/SwinUnetisic2018224/model_pretrain_epo100_bs16_lr0.05_imgsize224_s1234/best_model.pth')

net1.load_state_dict(ckpt_1['model_weights'])
net2.load_state_dict(ckpt_2['model_weights'])
net3.load_state_dict(ckpt_3['model_weights'])
net4.load_state_dict(ckpt_4['model_weights'])
print("Loaded checkpoints")


nets = []
nets.append(net1)
#nets.append(net2)
#nets.append(net3)
#nets.append(net4)

scores = []
score_1 = []
score_2 = []
score_3 = []
score_4 = []
scores.append(score_1)
#scores.append(score_2)
#scores.append(score_3)
#scores.append(score_4)

# Iterate through nets
for score, net in zip(scores, nets):
    predictions = []
    gt = []
    with torch.no_grad():
        net.cuda()
        net.eval()
        for itter,batch in tqdm(enumerate(test_loader)):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask']
            msk_pred = net(img)
            msk_np = msk.numpy()[0,0]
            gt.append(msk.numpy()[0, 0])
            msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
            msk_pred  = np.where(msk_pred>=0.5, 1, 0)
            msk_pred = binary_opening(msk_pred, structure=np.ones((2,3))).astype(msk_pred.dtype)
            msk_pred = binary_fill_holes(msk_pred, structure=np.ones((2,3))).astype(msk_pred.dtype)
            predictions.append(msk_pred)       
            
            y_scores = msk_pred.reshape(-1) #predictions.reshape(-1)
            y_true   = msk_np.reshape(-1) #gt.reshape(-1)

            y_scores2 = np.where(y_scores>0.47, 1, 0)
            y_true2   = np.where(y_true>0.5, 1, 0)
            
            F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)
            print ("\nF1 score (F-measure) or DSC: " +str(F1_score))
            score.append(F1_score)
        
    net.cpu()
    
# evaluate scores
# compare the list entries to each other
# Check if f1 of net 1 is bigger than the others
margin = 0.05
for i in range(len(scores[0])):
    #if (scores[0][i] > scores[1][i]) and (scores[0][i] > scores[2][i]) and (scores[0][i] > scores[3][i]):
    if (scores[0][i] < 0.7):
        with open('bad_d_lka_results.txt', 'a') as f:
                f.write('Our model is bad for Image:'+ str(i+1) + '\n')
                
    #if (scores[0][i] > scores[1][i] + margin) and (scores[0][i] > scores[2][i] + margin) and (scores[0][i] > scores[3][i] + margin):
    #    with open('comparison_margin_results.txt', 'a') as f:
    #            f.write('Our model is best with a margin for Image:'+ str(i+1) + '\n')