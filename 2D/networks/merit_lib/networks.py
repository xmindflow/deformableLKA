import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import logging

from scipy import ndimage

from networks.merit_lib.decoders import CASCADE_Add, CASCADE_Cat
#from merit_lib.decoders import CASCADE_Add, CASCADE_Cat
from networks.merit_lib.maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from networks.merit_lib.maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from networks.merit_lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from networks.merit_lib.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out
#from merit_lib.maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
#from merit_lib.maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
#from merit_lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
#from merit_lib.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out


logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def load_pretrained_weights(img_size, model_scale):
    
    if(model_scale=='tiny'):
        if img_size==224:
            backbone = maxvit_tiny_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
        elif(img_size==256):
            backbone = maxvit_rmlp_tiny_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    elif(model_scale=='small'):
        if img_size==224:
            backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    else:
        sys.exit(model_scale+" is not a valid model scale! Currently supported model scales are 'tiny' and 'small'.")
        
    backbone.load_state_dict(state_dict, strict=False)
    print('Pretrain weights loaded.')
    
    return backbone

class MaxViT(nn.Module):
    def __init__(self, n_class=1, img_size=224, model_scale='small'):
        super(MaxViT, self).__init__()
        
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = load_pretrained_weights(img_size, model_scale)   
          
        if(model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(model_scale=='small'):
            self.channels = [768, 384, 192, 96]
        
        # Prediction heads initialization
        self.out_head = nn.Conv2d(self.channels[0], self.n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        f = self.backbone(x)
        
        #print([f.shape])
        
        # prediction heads  
        p = self.out_head(f[3])
      
        #print([p1.shape])
        
        p = F.interpolate(p, scale_factor=32, mode='bilinear')
       
        #print([p.shape])
        
        return p

class MaxViT_Small(nn.Module):
    def __init__(self, n_class=1, img_size=224):
        super(MaxViT_Small, self).__init__()
        
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if img_size==224:
            self.backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            self.backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
              
        self.backbone.load_state_dict(state_dict, strict=False)
        
        print('Pretrain weights loaded.')
        
        self.channels=[768, 384, 192, 96] #[512, 256, 128, 64]
        
        # Prediction heads initialization
        self.out_head = nn.Conv2d(self.channels[0], self.n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        f = self.backbone(x)
        
        # print([f.shape])        
        
        # prediction heads  
        p = self.out_head(f[3])

        #print([p.shape])
        
        p = F.interpolate(p, scale_factor=32, mode='bilinear')

        #print([p.shape])
        
        return p
        
class MaxViT4Out(nn.Module):
    def __init__(self, n_class=1, img_size=224, model_scale='small'):
        super(MaxViT4Out, self).__init__()
        
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight        
        self.backbone = load_pretrained_weights(img_size, model_scale)
        
        if(model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(model_scale=='small'):
            self.channels = [768, 384, 192, 96]       
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        f = self.backbone(x)
        
        #print([f[3].shape,f[2].shape,f[1].shape,f[0].shape])

        x1_o, x2_o, x3_o, x4_o = f[3], f[2], f[1], f[0]
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear') 
         
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        return p1, p2, p3, p4
        
class MaxViT4Out_Small(nn.Module):
    def __init__(self, n_class=1, img_size=224, pretrain=True):
        super(MaxViT4Out_Small, self).__init__()
        
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if img_size==224:
            self.backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
            if pretrain:
                print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
                state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            self.backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
        
        if pretrain:
            self.backbone.load_state_dict(state_dict, strict=False)
            print('Pretrain weights loaded.')
        
        self.channels=[768, 384, 192, 96] #[512, 256, 128, 64]
        
        # Prediction heads initialization
        #self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        #self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        #self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        #self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        f = self.backbone(x)
        
        #print([f[3].shape,f[2].shape,f[1].shape,f[0].shape])

        x1_o, x2_o, x3_o, x4_o = f[3], f[2], f[1], f[0]
        
        # prediction heads  
        #p1 = self.out_head1(x1_o)
        #p2 = self.out_head2(x2_o)
        #p3 = self.out_head3(x3_o)
        #p4 = self.out_head4(x4_o)

        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        #p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        #p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        #p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        #p4 = F.interpolate(p4, scale_factor=4, mode='bilinear') 
         
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        #return p1, p2, p3, p4
        return x1_o, x2_o, x3_o, x4_o

class MaxViT_CASCADE(nn.Module):
    def __init__(self, n_class=1, img_size=224, model_scale='small', decoder_aggregation='additive'):
        super(MaxViT_CASCADE, self).__init__()
        
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = load_pretrained_weights(img_size, model_scale)
        
        if(model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(model_scale=='small'):
            self.channels = [768, 384, 192, 96]
     
        # decoder initialization
        if(decoder_aggregation=='additive'):
            self.decoder = CASCADE_Add(channels=self.channels)
        elif(decoder_aggregation=='concatenation'):
            self.decoder = CASCADE_Cat(channels=self.channels)
        else:
            sys.exit("'"+decoder_aggregation+"' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        f = self.backbone(x)
        
        #print([f[3].shape,f[2].shape,f[1].shape,f[0].shape])
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(f[3], [f[2], f[1], f[0]])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        return p1, p2, p3, p4

class MaxViT_CASCADE_Small(nn.Module):
    def __init__(self, n_class=1, img_size=224):
        super(MaxViT_CASCADE_Small, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if img_size==224:
            self.backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            self.backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
        
        self.backbone.load_state_dict(state_dict, strict=False)
        print('Pretrain weights loaded.')
        
        #channels=[512, 256, 128, 64]
        channels = [768, 384, 192, 96]
        # decoder initialization
        if(decoder_aggregation=='additive'):
            self.decoder = CASCADE_Add(channels=channels)
        elif(decoder_aggregation=='concatenation'):
            self.decoder = CASCADE_Cat(channels=channels)
        else:
            sys.exit("'"+decoder_aggregation+"' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(channels[2], n_class, 1)

        self.out_head4 = nn.Conv2d(channels[3], n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        f = self.backbone(x)
        
        #print([f[3].shape,f[2].shape,f[1].shape,f[0].shape])
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(f[3], [f[2], f[1], f[0]])
        #x1_o, x2_o, x3_o, x4_o = f[3], f[2], f[1], f[0]
        #x1_o = f
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        return p1, p2, p3, p4

class MERIT_Parallel(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear'):
        super(MERIT_Parallel, self).__init__()
        
        self.n_class = n_class
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale
        self.decoder_aggregation = decoder_aggregation       
        self.interpolation = interpolation

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale)
        self.backbone2 = load_pretrained_weights(self.img_size_s2[0], self.model_scale)
        
        if(self.model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(self.model_scale=='small'):
            self.channels = [768, 384, 192, 96]
     
        # shared decoder initialization
        if(self.decoder_aggregation=='additive'):
            self.decoder = CASCADE_Add(channels=self.channels)
        elif(self.decoder_aggregation=='concatenation'):
            self.decoder = CASCADE_Cat(channels=self.channels)
        else:
            sys.exit("'"+self.decoder_aggregation+"' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        #print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])
        
        f2 = self.backbone2(F.interpolate(x, size=self.img_size_s2, mode=self.interpolation))
        #print([f2[3].shape,f2[2].shape,f2[1].shape,f2[0].shape])
        
        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])
        
        x21_o, x22_o, x23_o, x24_o = self.decoder(f2[3], [f2[2], f2[1], f2[0]])
        
        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)
        #print([p11.shape,p12.shape,p13.shape,p14.shape])
        
        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)
        #print([p21.shape,p22.shape,p23.shape,p24.shape])
        
        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)
        
        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        return p1, p2, p3, p4

class MERIT_Cascaded(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear'):
        super(MERIT_Cascaded, self).__init__()
        
        self.n_class = n_class
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale 
        self.decoder_aggregation = decoder_aggregation      
        self.interpolation = interpolation
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale)
        self.backbone2 = load_pretrained_weights(self.img_size_s2[0], self.model_scale)
        
        if(self.model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(self.model_scale=='small'):
            self.channels = [768, 384, 192, 96]
     
        # decoder initialization
        if(self.decoder_aggregation=='additive'):
            self.decoder = CASCADE_Add(channels=self.channels)
        elif(self.decoder_aggregation=='concatenation'):
            self.decoder = CASCADE_Cat(channels=self.channels)
        else:
            sys.exit("'"+self.decoder_aggregation+"' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")
            
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        #print(x.shape)
        
        # transformer backbone as encoder
        if(x.shape[2]%14!=0):
            f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        else:
            f1 = self.backbone2(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
               
        #print([f[3].shape,f[2].shape,f[1].shape,f[0].shape])
        
        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])

        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)
        
        # calculate feedback from 1st decoder
        p14_in = self.out_head4_in(x14_o)
        p14_in = self.sigmoid(p14_in)
        

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)
        
        #print([p11.shape,p12.shape,p13.shape,p14.shape])

        p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)
        
        # apply feedback from 1st decoder to input
        x_in = x * p14_in
        
        if(x.shape[2]%14!=0):
            f2 = self.backbone2(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
        else:
            f2 = self.backbone1(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
            
        skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)
        
        x21_o, x22_o, x23_o, x24_o = self.decoder(f2[3]+skip1_3, [f2[2]+skip1_2, f2[1]+skip1_1, f2[0]+skip1_0])

        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)

        #print([p21.shape,p22.shape,p23.shape,p24.shape])
              
        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        return p1, p2, p3, p4

                        
class MERIT_Parallel_Small(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), decoder_aggregation='additive', interpolation='bilinear'):
        super(MERIT_Parallel_Small, self).__init__()
        
        self.interpolation = interpolation
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        #self.backbone = maxvit_tiny_rw_224_4out()  # [64, 128, 320, 512]
        self.backbone1 = maxxvit_rmlp_small_rw_256_4out()  # [64, 128, 320, 512]
        self.backbone2 = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
        
        print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        state_dict1 = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')        
        self.backbone1.load_state_dict(state_dict1, strict=False)
        
        print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')       
        state_dict2 = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')        
        self.backbone2.load_state_dict(state_dict2, strict=False)
        
        print('Pretrain weights loaded.')
        
        #channels=[512, 256, 128, 64]
        channels = [768, 384, 192, 96]
        # decoder initialization
        if(decoder_aggregation=='additive'):
            self.decoder = CASCADE_Add(channels=channels)
        elif(self.decoder_aggregation=='concatenation'):
            self.decoder = CASCADE_Cat(channels=self.channels)
        else:
            sys.exit("'"+self.decoder_aggregation+"' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(channels[3], n_class, 1)

    def forward(self, x, im1_size=(256,256), im2_size=(224,224)):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        if(x.shape[2]%14!=0):
            f1 = self.backbone1(F.interpolate(x, size=im1_size, mode=self.interpolation))
            f2 = self.backbone2(F.interpolate(x, size=im2_size, mode=self.interpolation))
        else:
            f1 = self.backbone2(F.interpolate(x, size=im1_size, mode=self.interpolation))
            f2 = self.backbone1(F.interpolate(x, size=im2_size, mode=self.interpolation))

        #print([f[3].shape,f[2].shape,f[1].shape,f[0].shape])
        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])
        x21_o, x22_o, x23_o, x24_o = self.decoder(f2[3], [f2[2], f2[1], f2[0]])

        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)
        
        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)

        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)
        
        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        return p1, p2, p3, p4

class MERIT_Cascaded_Small(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), decoder_aggregation='additive', interpolation='bilinear'):
        super(MERIT_Cascaded_Small, self).__init__()
        
        self.interpolation = interpolation
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone1 = maxxvit_rmlp_small_rw_256_4out()  # [64, 128, 320, 512]
        self.backbone2 = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
        print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        state_dict1 = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')        
        self.backbone1.load_state_dict(state_dict1, strict=False)
        
        print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')        
        state_dict2 = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        self.backbone2.load_state_dict(state_dict2, strict=False)
        
        print('Pretrain weights loaded.')
        
        #channels=[512, 256, 128, 64]
        channels = [768, 384, 192, 96]
        # decoder initialization
        if(decoder_aggregation=='additive'):
            self.decoder = CASCADE_Add(channels=channels)
        elif(self.decoder_aggregation=='concatenation'):
            self.decoder = CASCADE_Cat(channels=self.channels)
        else:
            sys.exit("'"+self.decoder_aggregation+"' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(channels[3], n_class, 1)

        self.out_head4_in = nn.Conv2d(channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, im1_size=(256,256), im2_size=(224,224)):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        #print(x.shape)
        # transformer backbone as encoder
        #print([x.shape, im1_size])
        if(x.shape[2]%14!=0):
            f1 = self.backbone1(F.interpolate(x, size=im1_size, mode=self.interpolation))
        else:
            f1 = self.backbone2(F.interpolate(x, size=im1_size, mode=self.interpolation))
                      
        #print([f[3].shape,f[2].shape,f[1].shape,f[0].shape])
        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])
     
        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)

        p14_in = self.out_head4_in(x14_o)
        p14_in = self.sigmoid(p14_in)
        

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)

        p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)
        
        x_in = x * p14_in #+ x
        
        if(x.shape[2]%14!=0):
            f2 = self.backbone2(F.interpolate(x_in, size=im2_size, mode=self.interpolation))
        else:
            f2 = self.backbone1(F.interpolate(x_in, size=im2_size, mode=self.interpolation))
               
        skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)
        
        x21_o, x22_o, x23_o, x24_o = self.decoder(f2[3]+skip1_3, [f2[2]+skip1_2, f2[1]+skip1_1, f2[0]+skip1_0])

        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)

        #print([p1.shape,p2.shape,p3.shape,p4.shape])
              
        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        return p1, p2, p3, p4
        
if __name__ == '__main__':
    model = MERIT_Cascaded_Small().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())

