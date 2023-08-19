import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
	            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x) #torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    

class CASCADE_Cat(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Cat,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=2*channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2*channels[1])
        self.CA2 = ChannelAttention(2*channels[2])
        self.CA1 = ChannelAttention(2*channels[3])
        
        self.SA = SpatialAttention()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # Concat 3
        d3 = torch.cat((x3,d3),dim=1)
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # Concat 2
        d2 = torch.cat((x2,d2),dim=1)
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # Concat 1
        d1 = torch.cat((x1,d1),dim=1)
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
        

class CASCADE_Add(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Add,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SpatialAttention()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # aggregate 3
        d3 = d3 + x3
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # aggregate 2
        d2 = d2 + x2
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # aggregate 1
        d1 = d1 + x1
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
