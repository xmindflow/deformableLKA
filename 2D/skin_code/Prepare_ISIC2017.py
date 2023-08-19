# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019
@author: Reza Azad
"""
from __future__ import division
import numpy as np
import scipy.io as sio
#import scipy.misc as sc
import glob
from imageio.v2 import imread
from PIL import Image


# Parameters
height = 224
width  = 224
channels = 3

############################################################# Prepare ISIC 2017 data set #################################################
Dataset_add = './ISIC2017/'
Tr_add = 'ISIC-2017_Training_Data'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
# It contains 2594 training samples
Data_train_2017    = np.zeros([2000, height, width, channels])
Label_train_2017   = np.zeros([2000, height, width])

print('Reading ISIC 2017')
for idx in range(len(Tr_list)):
    print(idx+1)
    img = imread(Tr_list[idx])
    pil_img = Image.fromarray(img)
   
    img = np.double(pil_img.resize((height, width), Image.BILINEAR))
    Data_train_2017[idx, :,:,:] = img

    
    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'ISIC-2017_Training_Part1_GroundTruth/' + b +'_segmentation.png')    
    img2 = imread(add)
    pil_img2 = Image.fromarray(img2)
    img2 = np.double(pil_img2.resize((height, width), Image.BILINEAR))
    Label_train_2017[idx, :,:] = img2    
         
print('Reading ISIC 2017 finished')

################################################################ Make the train and test sets ########################################    
# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing

Train_img      = Data_train_2017[0:1399,:,:,:]
Validation_img = Data_train_2017[1399:1399+200,:,:,:]
Test_img       = Data_train_2017[1399+200:1999,:,:,:]

Train_mask      = Label_train_2017[0:1399,:,:]
Validation_mask = Label_train_2017[1399:1399+200,:,:]
Test_mask       = Label_train_2017[1399+200:1999,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)


