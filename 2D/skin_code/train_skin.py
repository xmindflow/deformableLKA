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
import copy
import yaml
from model.vit_seg_modeling import VisionTransformer as ViT_seg
from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.MaxViT_deform_LKA import MaxViT_deformableLKAFormer
import sys

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
args = parser.parse_args()
#args = parser.parse_args(args=[])


## Loader
## Hyper parameters
config         = yaml.load(open('./config_skin.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: {}".format(device))
data_path = args.root_path

train_dataset = isic_loader(path_Data = data_path, train = True)
train_loader  = DataLoader(train_dataset, batch_size = int(args.batch_size), shuffle= True)
val_dataset   = isic_loader(path_Data = data_path, train = False)
val_loader    = DataLoader(val_dataset, batch_size = 1, shuffle= False)
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
args.exp = 'MaxViT_' + dataset_name + str(args.img_size)
snapshot_path = "./model_results2018/{}/{}".format(args.exp, 'model')
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
    
Net = MaxViT_deformableLKAFormer(img_size=args.img_size, num_classes=1).to(device)

optimizer = optim.SGD(Net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 10)
#criteria  = torch.nn.BCELoss()
criteria = torch.nn.BCEWithLogitsLoss()
print("Created net and optimizers.")
print("Start training...")

for ep in range(int(args.max_epochs)):
    print("Current epoch: {}".format(ep))
    Net.train()
    epoch_loss = 0
    for itter, batch in enumerate(train_loader):
        img = batch['image'].to(device, dtype=torch.float)
        #print("Image shape: {}".format(img.shape))
        msk = batch['mask'].to(device)
        mask_type = torch.float32
        msk = msk.to(device=device, dtype=mask_type)
        msk_pred = Net(img)
        loss = criteria(msk_pred, msk) 
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()  
        if itter%int(float(config['progress_p']) * len(train_loader))==0:
            print(f' Epoch>> {ep+1} and itteration {itter+1} Loss>> {((epoch_loss/(itter+1)))}')
    ## Validation phase
    with torch.no_grad():
        print('val_mode')
        val_loss = 0
        Net.eval()
        for itter, batch in enumerate(val_loader):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            mask_type = torch.float32
            msk = msk.to(device=device, dtype=mask_type)
            msk_pred = Net(img)
            loss = criteria(msk_pred, msk) 
            val_loss += loss.item()
        print(f' validation on epoch>> {ep+1} dice loss>> {(abs(val_loss/(itter+1)))}')     
        mean_val_loss = (val_loss/(itter+1))
        # Check the performance and save the model
        if (mean_val_loss) < best_val_loss:
            print('New best loss, saving...')
            best_val_loss = copy.deepcopy(mean_val_loss)
            state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})
            torch.save(state, snapshot_path + '/best_model.pth')

    scheduler.step(mean_val_loss)
    
print('Trainng phase finished')    