'''
Created on Dec 10, 2021

@author: blew
'''

import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
# import os
# import sys
import numpy as np


class ResidBlock(nn.Module):
    def __init__(self,Nin, Nout, NCh=3,stride=1):
        '''
        Nin - number of input channels
        Nout - number of output channels from the block
        NCh - number of inner channels (or number of output channels
            from the first Conv1d layer)
            
        stride - stride param at first conv1d
        '''
        super(ResidBlock, self).__init__()
        kernel_size=3
        pad=kernel_size//2
        self.conv1=nn.Conv1d(in_channels=Nin, out_channels=NCh, 
                             kernel_size=kernel_size,stride=stride, padding=pad)
        
        self.conv2=nn.Conv1d(in_channels=NCh, out_channels=Nout, 
                             kernel_size=1,stride=1)

        self.bn1=nn.BatchNorm1d(num_features=Nout)

        self.downsample=nn.Sequential(
            nn.Conv1d(Nin,Nout,kernel_size,stride=stride,padding=pad),
            self.bn1,
            )
        # if stride==1:
        #     self.downsample=nn.Identity()
        # else:
        # self.downsample=nn.AvgPool1d(kernel_size=3,stride=stride, padding='same')
        
    def forward(self,x):
        y=self.conv1(x)
        y=self.bn1(y)
        y=F.relu(y)
        y=self.conv2(y)
        
        y=self.downsample(x)+y
        y=F.relu(y)
        return y

class ResidPart(nn.Module):
    def __init__(self, Nblk,Nin,Nout,NCresidBlk,stride=1):
        '''
        Nin, Nout, NCresidBlk, stride - passed to residual block
        
        Nblk - number of residual blocks in part
        '''
        super(ResidPart, self).__init__()
        self.part=nn.ModuleList()
        for i in range(Nblk):
            if i==0:
                self.part.append(ResidBlock(Nin,Nout,NCh=NCresidBlk,stride=stride))
            else:
                self.part.append(ResidBlock(Nout,Nout,NCh=NCresidBlk,stride=1))

    def forward(self,x):
        for i, b in enumerate(self.part):
            # print('{} x.shape: {}'.format(i,x.shape))
            x=b(x)
            # print('{} y.shape: {}'.format(i,x.shape))
        return x

class ResidNN(nn.Module):
    def __init__(self,dsize,Nin=1,part_chs=[16,32,64],Nblk_per_part=3,stride=None):
        super(ResidNN,self).__init__()
        
        self.stride=np.ones(len(part_chs),dtype=int)*2
        self.stride[0]=1
        if stride!=None:
            self.stride=np.array(stride,dtype=int)

        self.Nin=Nin
        self.dsize=dsize
        self.Nblk=Nblk_per_part
        downscale=self.stride.prod()
        
        self.net=nn.ModuleList()
        for i,pc in enumerate(part_chs):
            if i==0:
                self.net.append(ResidPart(self.Nblk,self.Nin,pc,pc,stride=self.stride[i]))
            else:
                self.net.append(ResidPart(self.Nblk,part_chs[i-1],pc,pc,stride=self.stride[i]))
        
        # self.pool=nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool=nn.AdaptiveAvgPool1d(dsize*downscale)
        self.lastFC=nn.Sequential(
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Linear(dsize*downscale,dsize),
            )
              
    def forward(self,x):
        for i, p in enumerate(self.net):
            # print('part: ',i)
            # print('in: ',x.shape)
            x=p(x)
            # print('out: ',x.shape)
            
        x=x.view(x.shape[0],1,-1)
        # print('before pool: ',x.shape)
        x=self.pool(x)
        # print('out pool: ',x.shape)
        x=self.lastFC(x)
        x=F.softmax(x, dim=-1)
        return x
        

class Conv1(nn.Module):
    def __init__(self, conf=[1024,512,256], nntype='conv1d'):
        '''
        '''
        super().__init__()
        self.nntype=nntype
        self.conf=conf

        self.fc=nn.ModuleList()
 
        for i,s in enumerate(conf):
            if i>0:
                self.fc.append(nn.Linear(conf[i-1], s))
                self.fc.append(nn.Dropout(p=0.1))

        C=64
        k=64
        self.conv1=nn.Conv1d(1,C,k)
        self.init_network()

    def init_network(self):
        for L in self.fc:
            # L.weight.data.uniform_(0.0, 1.0)
            if isinstance(L,nn.Linear):
                L.bias.data.fill_(0.0)
                nn.init.xavier_uniform_(L.weight)
    
        
        
    def forward(self,x):
        n=len(self.fc)
        x = x.view(x.size(0),1, -1)
        x=F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        for i,fc in enumerate(self.fc,1):
            if self.nntype=='conv1d':
                if i<n:
                    x = F.relu(fc(x))


        x = x.view(x.size(0),1, -1)
        x=F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)

        lastLinear=nn.Linear(x.size(-1),self.conf[0]).to(x.device)
        x = F.relu(lastLinear(x))
                
        return x,None
        
        

# class DenseAutoencoder(DenseFF):
#     def __init__(self,input_size, min_size):
#         hl=[]
#         hs=input_size
#         # encoder
#         while hs>min_size:
#             hl.append(hs)
#             hs=hs//2
#
#         # decoder
#
#         super().__init__(size=hl)

class DenseFF(nn.Module):
    '''
    FC ANN model 
    '''


    def __init__(self, size=[120,240,64,3], nntype='class', **kwargs):
        '''
        Constructor
        '''
        super().__init__()
        self.nntype=nntype
        self.dropout_rate=0.1
        if 'dropout' in kwargs.keys():
            self.dropout_rate=kwargs['dropout']

        self.fc=nn.ModuleList()
 
        for i,s in enumerate(size):
            if i>0:
                self.fc.append(nn.Linear(size[i-1], s))
                if self.dropout_rate>0:
                    self.fc.append(nn.Dropout(p=self.dropout_rate))

        self.init_network()

        self.last_linear=self.fc[-1]
#         self.fc1=nn.Linear(120,240)
#         self.fc2=nn.Linear(240,64)
#         self.fc3=nn.Linear(64,3)

#         self.fc1=nn.Linear(60,30)
#         self.fc2=nn.Linear(30,30)
#         self.fc3=nn.Linear(30,3)

    def init_network(self):
        for L in self.fc:
            # L.weight.data.uniform_(0.0, 1.0)
            if isinstance(L,nn.Linear):
                L.bias.data.fill_(0.0)
                nn.init.xavier_uniform_(L.weight)

    def forward(self, x):
        n=len(self.fc)
        # x = F.softmax(x,dim=-1) # don't relu on the last layer
        for i,fc in enumerate(self.fc,1):
            if self.nntype=='class':
                if i<n:
                    x = F.relu(fc(x))
                else:
                    x = F.softmax(fc(x),dim=-1) # don't relu on the last layer
            elif self.nntype=='autoenc':
                x = F.relu(fc(x))
            elif self.nntype=='dense':
                x = F.relu(fc(x))
                
        return x,None
        
#         x=F.relu(self.fc1(x))
#         x=F.relu(self.fc2(x))
#         x=F.softmax(self.fc3(x))
#         x=self.fc3(x)
#         return x
        
        
        
    def parameters(self, only_trainable=True):
        for param in self.fc.parameters():
            if only_trainable and not param.requires_grad:
                continue
            yield param        
        
    def get_class_num(self):
        return len(self.last_linear.weight)
