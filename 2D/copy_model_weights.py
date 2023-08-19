## file to copy the weights from layer one to layer two.

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.MaxViT_deform_LKA import  MaxViT_deformableLKAFormer
from utils import test_single_volume
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

from torch.utils.data import DataLoader

from tqdm import tqdm

def inference(model, testloader, args, test_save_path=None):
        model.eval()
        metric_list = 0.0

        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=9, patch_size=[224, 224],
                                        test_save_path=test_save_path, case=case_name, z_spacing=3)
            metric_list += np.array(metric_i)
            
        
        metric_list = metric_list / len(testloader.dataset)
        
        for i in range(1, 9):
            print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

if __name__ == "__main__":
    
    
   
    
    
    net = MaxViT_deformableLKAFormer().cuda(0)
    #print(net)
    
    # Load network state dict
    
    # model_data = torch.load('/work/scratch/niggemeier/projects/Cluster_BiDAEFormer/model_out/MaxViT_deform_LKA_one_layer/MaxViT_deform_LKA_seed_1234_epoch_359.pth')
    
    # net.load_state_dict(model_data)
    
    # #net2 = MaxViT_deformableLKAFormer()
    # #net2.decoder_0.layer_lka_2.copy_(net.decoder_0.layer_lka_1)
    
    # print("Copying layer weights")
    # net.decoder_0.layer_lka_2.load_state_dict(net.decoder_0.layer_lka_1.state_dict())
    # net.decoder_1.layer_lka_2.load_state_dict(net.decoder_1.layer_lka_1.state_dict())
    # net.decoder_2.layer_lka_2.load_state_dict(net.decoder_2.layer_lka_1.state_dict())
    # net.decoder_3.layer_lka_2.load_state_dict(net.decoder_3.layer_lka_1.state_dict())
    # print("Done.")
    
    # # Save new weights
    # torch.save(net.state_dict(), '/work/scratch/niggemeier/projects/Cluster_BiDAEFormer/model_out/MaxViT_deform_LKA_one_layer/MaxViT_deform_LKA_seed_1234_epoch_359_copied_layers_test.pth')
    
    net.load_state_dict(torch.load('/work/scratch/niggemeier/projects/Cluster_BiDAEFormer/model_out/MaxViT_deform_LKA_one_layer/MaxViT_deform_LKA_seed_1234_epoch_359_copied_layers_test.pth'))
    
    db_test = Synapse_dataset(base_dir="./data/Synapse/test_vol_h5", split="test_vol", list_dir="./lists/lists_Synapse", img_size=224)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    mean_dice, mean_hd95 = inference(net, testloader, None, test_save_path='/work/scratch/niggemeier/projects/Cluster_BiDAEFormer/visualization/3d_nii_d_lka_former')