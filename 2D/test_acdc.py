import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils2 import test_single_volume
from dataset_ACDC import ACDCdataset, RandomGenerator
from networks.MaxViT_deform_LKA import MaxViT_deformableLKAFormerTrEcaGanorm
# from lib.networks import TransCASCADE, PVT_CASCADE
# from lib.cnn_vit_backbone import CONFIGS as CONFIGS_ViT_seg


def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
            np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        #mean_jacard = np.mean(metric_list, axis=0)[2]
        #mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info(
            'Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        logging.info("Testing Finished!")
        return performance, mean_hd95


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=12, help="batch size")
    parser.add_argument("--lr", default=0.0001, help="learning rate")
    parser.add_argument("--max_epochs", default=150)
    parser.add_argument("--img_size", default=224)
    parser.add_argument("--save_path", default="./model_pth/ACDC")
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
    parser.add_argument("--root_dir", default="./data/ACDC/")
    parser.add_argument("--volume_path", default="./data/ACDC/test")
    parser.add_argument("--z_spacing", default=10)
    parser.add_argument("--num_classes", default=4)
    parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
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
    args.exp = 'TransCASCADE_' + str(args.img_size)
    snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'TransCASCADE')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    #snapshot_path += '_' + args.vit_name
    #snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    #snapshot_path = snapshot_path + '_vitpatch' + str(
    #    args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    # config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    # if args.vit_name.find('R50') != -1:
    #     config_vit.patches.grid = (
    #     int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = MaxViT_deformableLKAFormerTrEcaGanorm(num_classes= args.num_classes).cuda(0) #TODO: Image size should be checked
    #net = TransCASCADE(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # net = PVT_CASCADE(n_class=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best', 'epoch_' + str(args.max_epochs - 1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
    test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
    os.makedirs(test_save_path, exist_ok=True)

    db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    results = inference(args, net, testloader, test_save_path)
