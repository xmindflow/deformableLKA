from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation

# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


## Temporary
class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False):
        super(isic_loader, self)
        self.train = train
        if train:
          self.data   = np.load(path_Data+'data_train.npy')
          self.mask   = np.load(path_Data+'mask_train.npy')
        else:
          if Test:
            self.data   = np.load(path_Data+'data_test.npy')
            self.mask   = np.load(path_Data+'mask_test.npy')
          else:
            self.data   = np.load(path_Data+'data_val.npy')
            self.mask   = np.load(path_Data+'mask_val.npy')          
          
          
        self.data   = dataset_normalized(self.data)
        self.mask   = np.expand_dims(self.mask, axis=3)
        self.mask   = self.mask /255.

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.train:
            img, seg = self.apply_augmentation(img, seg)
        
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute( 2, 0, 1)
        seg = seg.permute( 2, 0, 1)

        return {'image': img,
                'mask' : seg}
               
    def apply_augmentation(self, img, seg):
        if random.random() < 0.5:
            img  = np.flip(img,  axis=1)
            seg  = np.flip(seg,  axis=1)
        return img, seg

    def __len__(self):
        return len(self.data)
    