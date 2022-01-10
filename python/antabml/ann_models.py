'''
Created on Dec 10, 2021

@author: blew
'''

import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
# import os
# import sys

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


    def __init__(self, size=[120,240,64,3], nntype='class'):
        '''
        Constructor
        '''
        super().__init__()
        self.nntype=nntype

        self.fc=nn.ModuleList()
 
        for i,s in enumerate(size):
            if i>0:
                self.fc.append(nn.Linear(size[i-1], s))

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
            L.bias.data.fill_(0.0)
            nn.init.xavier_uniform(L.weight)

    def forward(self, x):
        n=len(self.fc)
        x = F.softmax(x,dim=-1) # don't relu on the last layer
        for i,fc in enumerate(self.fc,1):
            if self.nntype=='class':
                if i<n:
                    x = F.relu(fc(x))
                else:
                    x = F.softmax(fc(x),dim=-1) # don't relu on the last layer
            elif self.nntype=='autoenc':
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
