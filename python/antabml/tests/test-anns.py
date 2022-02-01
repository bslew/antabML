'''
Created on Jan 28, 2022

@author: blew
'''
import unittest
from .. import ann_models
import torch

def getData(N,C,L):
    x=torch.rand(N*C*L).view(N,C,L)
    return x

class Test(unittest.TestCase):

    def test_01_ResidBlk1(self):
        print('test ResidBlk')
        x=getData(2,1,10)
        blkmodel=ann_models.ResidBlock(1,3,3,stride=1)
        print(x.shape)
        y=blkmodel(x)
        print(y.shape)

    def test_01_ResidBlk2(self):
        print('test ResidBlk2')
        x=getData(2,1,10)
        blkmodel=ann_models.ResidBlock(1,3,3,stride=2)
        print(x.shape)
        y=blkmodel(x)
        print(y.shape)

    def test_02_ResidPart(self):
        print('test ResidPart')
        N=2
        C=1
        L=10
        x=getData(N,C,L)
        m=ann_models.ResidPart(6,C,3,3)
        print(x.shape)
        y=m(x)
        print(y.shape)
    
    def test_03_ResidConv1d(self):
        print('test ResidANN')
        N=2
        C=1
        L=5
        x=getData(N,C,L)
        m=ann_models.ResidNN(L, part_chs=[16,32,64,128])
        print(x.shape)
        y=m(x)
        print(y.shape)
        print(y)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testResidConv1d']
    unittest.main()