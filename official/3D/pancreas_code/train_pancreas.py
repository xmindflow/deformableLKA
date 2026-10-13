import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from networks.unetr import UNETR
from networks.d_lka_former.d_lka_net_synapse import D_LKA_Net
from networks.d_lka_former.transformerblock import TransformerBlock_3D_single_deform_LKA, TransformerBlock
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./dataset_pancreas', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--exp', type=str,  default="pancreas1", help='model_name')                               # todo model name
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train') # 6000
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + "/"

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size =(96,96,96)  # 96x96x96 for Pancreas
T = 0.1
Good_student = 0 # 0: vnet 1:resnet

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c-1):
        temp_line = vec[:,i,:].unsqueeze(1)  # b 1 c
        star_index = i+1
        rep_num = c-star_index
        repeat_line = temp_line.repeat(1, rep_num,1)
        two_patch = vec[:,star_index:,:]
        temp_cat = torch.cat((repeat_line,two_patch),dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result,dim=1)
    return  result

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    #if os.path.exists(snapshot_path + '/code'):
    #    shutil.rmtree(snapshot_path + '/code')
    #shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name ='dlka_former'):
        # Network definition
        if name == 'dlka_former':
            net = D_LKA_Net(in_channels=1, 
                           out_channels=num_classes, 
                           img_size=[96, 96, 96],
                           patch_size=(2,2,2),
                           input_size=[48*48*48, 24*24*24,12*12*12,6*6*6],
                           trans_block=TransformerBlock_3D_single_deform_LKA,
                           do_ds=False)
            model = net.cuda()
        return model

    model_d_lka_former = create_model(name='dlka_former')

    db_train = LAHeart(base_dir=train_data_path, split='train', train_flod='train0.list', common_transform=transforms.Compose([RandomCrop(patch_size),]),
                        sp_transform=transforms.Compose([ToTensor(),]))


    trainloader = DataLoader(db_train, batch_sampler=None, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    d_lka_former_optimizer = optim.SGD(model_d_lka_former.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr

    model_d_lka_former.train()

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{}, i_batch:{}'.format(epoch_num,i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']
            # Transfer to GPU
            lka_input,lka_label = volume_batch1.cuda(), volume_label1.cuda()

            # Network forward
            lka_outputs = model_d_lka_former(lka_input)

            ## calculate the supervised loss           
            lka_loss_seg = F.cross_entropy(lka_outputs[:labeled_bs], lka_label[:labeled_bs])
            lka_outputs_soft = F.softmax(lka_outputs, dim=1)
            lka_loss_seg_dice = losses.dice_loss(lka_outputs_soft[:labeled_bs, 1, :, :, :], lka_label[:labeled_bs] == 1)
            
            loss_total = lka_loss_seg + lka_loss_seg_dice
            # Network backward
            d_lka_former_optimizer.zero_grad()
            loss_total.backward()
            d_lka_former_optimizer.step()

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/total_loss', loss_total, iter_num)
            writer.add_scalar('loss/lka_loss_seg', lka_loss_seg, iter_num)
            writer.add_scalar('loss/lka_loss_seg_dice', lka_loss_seg_dice, iter_num)

            if iter_num % 50 == 0 and iter_num !=0:
                logging.info(
                    'iteration: %d Total loss : %f CE loss : %f Dice loss : %f'  %
                    (iter_num, loss_total.item(), lka_loss_seg.item(), lka_loss_seg_dice.item(),))

            ## change lr
            if iter_num % 2500 == 0 and iter_num!= 0:
                lr_ = lr_ * 0.1
                for param_group in d_lka_former_optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    save_mode_path_lka_net = os.path.join(snapshot_path, 'd_lka_former_iter_' + str(max_iterations) + '.pth')
    torch.save(model_d_lka_former.state_dict(), save_mode_path_lka_net)
    logging.info("save model to {}".format(save_mode_path_lka_net))

    writer.close()
