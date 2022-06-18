import torch
from torch import nn
import numpy as np
import pandas as pd
import warnings
import math

warnings.filterwarnings("ignore", category=UserWarning)

class AttentionBlock(nn.Module):
    def __init__(self,in_channels,idx,channel_attention = False,special_attention = False,
                 temporal_attention = False,spatial_attention = False):
        super(AttentionBlock,self).__init__()
        self.in_channels = in_channels
        self.idx = idx
        self.channel_attention = channel_attention
        self.special_attention = special_attention
        self.temporal_attention = temporal_attention
        self.spatial_attention = spatial_attention
        if self.special_attention:
            self.bt = 5
        if self.temporal_attention:
            self.bt = 8
        self.h = int(16//math.pow(2,self.idx))
        self.w = int(16//math.pow(2,self.idx))
        self.cha_atten = nn.Sequential( # 通道注意力
            nn.Linear(self.in_channels,self.in_channels),
            nn.Softmax()
        )
        self.spa_atten = nn.Sequential( # 空间注意力
            nn.Linear(self.h*self.w,self.h*self.w),
            nn.Softmax()
        )
        self.st_atten =nn.Sequential(   # 时/频注意力
            nn.Linear(self.bt,self.bt),
            nn.Softmax()
        )
        
    def forward(self,x): # (N,C,B/T,H,W)
        if self.channel_attention:
            atten1 = x.mean(4).mean(3).mean(2) # (N,C,1,1,1)
            atten1 = atten1.flatten(start_dim=1) # (N,C)
            atten1 = self.cha_atten(atten1)
            atten1 = atten1.unsqueeze(2).unsqueeze(3).unsqueeze(4) # (N,C,1,1,1)
            x = x * atten1
        if self.spatial_attention:
            atten2 = x.mean(2).mean(1) # (N,1,1,H,W)
            atten2 = atten2.flatten(start_dim=1) # (N,H*W)
            atten2 = self.spa_atten(atten2)
            atten2 = atten2.reshape(-1,self.h,self.w).unsqueeze(1).unsqueeze(2) # (N,1,1,H,W)
            x = x * atten2
        if self.special_attention or self.temporal_attention:
            atten3 = x.mean(4).mean(3).mean(1) # (N,1,B/T,1,1)
            atten3 = atten3.flatten(start_dim=1) # (N,B/T)
            atten3 = self.st_atten(atten3)
            atten3 = atten3.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [N,1,B/T,1,1]
            x = x * atten3
        return x # (N.C.B/T,H.W)
    

class ResUnit(nn.Module):
    def __init__(self,in_channels,rate):
        super(ResUnit,self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU()
        self.rate = rate
        self.botneck = nn.Sequential( # depth wise convolution
            nn.Conv3d(in_channels=in_channels,out_channels=int(self.rate * in_channels),kernel_size=(1,1,1),padding=(0,0,0),bias=False),
            nn.BatchNorm3d(int(self.rate * in_channels)),
            nn.ReLU(),
            nn.Conv3d(in_channels=int(self.rate * in_channels),out_channels=int(self.rate * in_channels),kernel_size=(3,3,3),padding=(1,1,1),groups=int(self.rate * in_channels),bias=False),
            nn.Conv3d(in_channels=int(self.rate * in_channels),out_channels=in_channels,kernel_size=(1,1,1),padding=(0,0,0),bias=False)
        )
        
    def forward(self,x):
        out = self.bn(x) 
        out = self.relu(out)
        out = self.botneck(out)
        out = torch.cat([x,out],1)
        return out

    
class ResBlock(nn.Module):
    def __init__(self,in_channels,rate,num_layer):
        super(ResBlock,self).__init__()
        self.layer = self._make_layer(in_channels,rate,num_layer)
        
    def _make_layer(self,in_channels,rate,num_layer):
        layers = []
        for i in range(num_layer):
            layers.append(ResUnit(in_channels,rate))
            in_channels = in_channels * 2
        return nn.Sequential(*layers)
        
    def forward(self,x):
        return self.layer(x)


class ResNet(nn.Module):
    def __init__(self,in_channels = 1,out_channels = 4,num_layers = 2,rate = 1,compression = 0.5,
                 channel_attention = False,special_attention = False,temporal_attention = False,spatial_attention = False):
        super(ResNet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_attention = channel_attention
        self.special_attention = special_attention
        self.temporal_attention = temporal_attention
        self.spatial_attention = spatial_attention
        # conv
        if special_attention:# 频域注意力 (16,1,5,32,32)→(16,12,5,16,16)
            self.conv1 = nn.Conv3d(self.in_channels,self.out_channels,kernel_size=(3,5,5),stride=(1,2,2),padding=(1,2,2),bias=False)
        if temporal_attention:# 时域注意力 (16,1,8,32,32)→(16,12,8,16,16)
            self.conv1 = nn.Conv3d(self.in_channels,self.out_channels,kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1),bias=False)
        self.in_channels = self.out_channels
        # block1
        self.idx = 0
        self.block1 = ResBlock(self.in_channels,rate,num_layers)
        self.in_channels = int(self.in_channels * math.pow(2,num_layers))
        self.trans1 = nn.Sequential(
            nn.Conv3d(self.in_channels,int(self.in_channels * compression),kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=False),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.in_channels = int(self.in_channels * compression)
        
        # block2
        self.idx = 1
        self.block2 = ResBlock(self.in_channels,rate,num_layers)
        self.in_channels = int(self.in_channels * math.pow(2,num_layers))
        self.trans2 = nn.Sequential(
            nn.Conv3d(self.in_channels,int(self.in_channels * compression),kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=False),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.in_channels = int(self.in_channels * compression)
        
        # block3
        self.idx = 2
        self.atten3 = AttentionBlock(self.in_channels,self.idx,
                                     self.channel_attention,self.special_attention,self.temporal_attention,self.spatial_attention)
        self.block3 = ResBlock(self.in_channels,rate,num_layers)
        self.in_channels = int(self.in_channels * math.pow(2,num_layers))
        self.trans3 = nn.Sequential(
            nn.Conv3d(self.in_channels,int(self.in_channels * compression),kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=False),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.in_channels = int(self.in_channels * compression)
        
        # block4
        self.idx = 3
        self.atten4 = AttentionBlock(self.in_channels,self.idx,
                                     self.channel_attention,self.special_attention,self.temporal_attention,self.spatial_attention)
        self.block4 = ResBlock(self.in_channels,rate,num_layers)
        self.in_channels = int(self.in_channels * math.pow(2,num_layers))
        self.trans4 = nn.Sequential(
            nn.Conv3d(self.in_channels,int(self.in_channels * compression),kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=False),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.in_channels = int(self.in_channels * compression)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.trans3(self.block3(self.atten3(out)))
        out = self.trans4(self.block4(self.atten4(out)))
        return out

class SSTCNN(nn.Module):
    def __init__(self):
        super(SSTCNN,self).__init__()
        self.in_channels = 1
        self.out_channels = 2
        self.num_layers = 2
        self.rate = 8.0
        self.compression = 0.5
        self.num = int(self.out_channels * math.pow(math.pow(2,self.num_layers) * self.compression,4) * 2)
        self.ss = ResNet(self.in_channels,self.out_channels,self.num_layers,self.rate,self.compression,
                         channel_attention = True,special_attention = True,temporal_attention = False,spatial_attention = True)
        self.st = ResNet(self.in_channels,self.out_channels,self.num_layers,self.rate,self.compression,
                         channel_attention = True,special_attention = False,temporal_attention = True,spatial_attention = True)
        self.avg = nn.AdaptiveAvgPool3d((1,1,1))
        self.cls = nn.Sequential(
            nn.Linear(self.num,50),
            nn.Dropout(0.5),
            nn.Linear(50,3)
        )
        
    def forward(self,x_ss,x_st):
        x1 = self.ss(x_ss)
        x1 = self.avg(x1)
        x2 = self.st(x_st)
        x2 = self.avg(x2)
        x1 = x1.view(-1, x1.shape[1] * x1.shape[2] * x1.shape[3] * x1.shape[4])
        x2 = x2.view(-1, x2.shape[1] * x2.shape[2] * x2.shape[3] * x2.shape[4])
        x = torch.cat([x1,x2],1)
        x = self.cls(x)
        return x
        