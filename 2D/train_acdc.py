import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random

import numpy as np
from tqdm import tqdm
from medpy.metric import dc, hd95
from tensorboardX import SummaryWriter
from utils2 import DiceLoss, calculate_dice_percase, val_single_volume, BoundaryDoULoss
from dataset_ACDC import ACDCdataset, RandomGenerator
from test_acdc import inference
from networks.MaxViT_deform_LKA import MaxViT_deformableLKAFormerTrEcaGanorm
# from lib.cnn_vit_backbone import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=12, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=150)
parser.add_argument("--img_size", default=224)
parser.add_argument("--save_path", default="./model_pth/ACDC") # Check the root Dir: SAVE path (checked)
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/") # Check the root Dir: Dataset root (Aval)
parser.add_argument("--volume_path", default="./data/ACDC/test") # Check the root Dir: test root (Aval)
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!') # Check the test_save Dir: Preds root
parser.add_argument("--model_name", type=str, default="MaxViT_deform_ECAGaNorm", help="model_name")
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# config_vit = CONFIGS_ViT_seg[args.vit_name]
# config_vit.n_classes = args.num_classes
# config_vit.n_skip = args.n_skip

args.is_pretrain = True
args.exp = 'EfCSViT_' + str(args.img_size)
snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'EfCSViT') # Save_path == Lrz/reza/sina
#snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
#snapshot_path += '_' + args.vit_name
#snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
#snapshot_path = snapshot_path + '_vitpatch' + str(
#    args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.exp)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

# if args.vit_name.find('R50') != -1:
#     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
#
# net = TransCASCADE(config_vit, img_size=args.img_size,
#                    num_classes=config_vit.n_classes).cuda()  # model initialization for TransCASCADE
# net.load_from(weights=np.load(config_vit.pretrained_path))
if args.model_name == "MaxViT_deform_ECAGaNorm":

    net = MaxViT_deformableLKAFormerTrEcaGanorm(num_classes= args.num_classes).cuda(0)

# net = PVT_CASCADE(n_class=config_vit.n_classes).cuda() # model initialization for PVT-CASCADE. comment above two lines if use PVT-CASCADE


if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
transforms.Compose(
    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val = ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader = DataLoader(db_val, batch_size=1, shuffle=False)
db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    net = nn.DataParallel(net)

net = net.cuda()
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
boundary_loss = BoundaryDoULoss(args.num_classes)
save_interval = args.n_skip

iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.5

logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
writer = SummaryWriter(snapshot_path + '/log')

# optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

def val():
    logging.info("Validation ===>")
    dc_sum = 0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(
            torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)
        #p1, p2, p3, p4 = net(val_image_batch) #change this part
        val_outputs = net(val_image_batch)
        # = p1 + p2 + p3 + p4

        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)

        dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

    print("val avg_dsc: %f" % (performance))
    return performance


for epoch in iterator:
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = net(image_batch)  # forward

        #outputs = p1 + p2 + p3 + p4  # additive output aggregation


        # loss_ce1 = ce_loss(p1, label_batch[:].long())
        # loss_ce2 = ce_loss(p2, label_batch[:].long())
        # loss_ce3 = ce_loss(p3, label_batch[:].long())
        # loss_ce4 = ce_loss(p4, label_batch[:].long())
        # loss_dice1 = dice_loss(p1, label_batch, softmax=True)
        # loss_dice2 = dice_loss(p2, label_batch, softmax=True)
        # loss_dice3 = dice_loss(p3, label_batch, softmax=True)
        # loss_dice4 = dice_loss(p4, label_batch, softmax=True)
        #
        # loss_p1 = 0.3 * loss_ce1 + 0.7 * loss_dice1
        # loss_p2 = 0.3 * loss_ce2 + 0.7 * loss_dice2
        # loss_p3 = 0.3 * loss_ce3 + 0.7 * loss_dice3
        # loss_p4 = 0.3 * loss_ce4 + 0.7 * loss_dice4
        #
        # alpha, beta, gamma, zeta = 1., 1., 1., 1.
        # loss = alpha * loss_p1 + beta * loss_p2 + gamma * loss_p3 + zeta * loss_p4  # current setting is for additive aggregation.

        loss = boundary_loss(outputs, label_batch[:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # We did not use this
        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/boundary_loss', loss, iter_num)

        #train_loss += loss.item()
        if iter_num % 20 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            # acc_loss_bo = acc_loss_bo / 100
        train_loss += loss.item()
    Loss.append(train_loss / len(train_dataset))
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))

    avg_dcs = val()

    if avg_dcs > Best_dcs:
        save_model_path = os.path.join(snapshot_path, 'best.pth')
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))

        Best_dcs = avg_dcs

        avg_dcs, avg_hd = inference(args, net, testloader, args.test_save_dir) #TODO: Check the Tests Output
        print("test avg_dsc: %f" % (avg_dcs))
        Test_Accuracy.append(avg_dcs)

    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path, 'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        iterator.close()
        break