'''
Created on Dec 10, 2021

@author: blew
'''
import torch
import torch.nn as nn
import numpy as np

def select_top_N(outputs, N):
    '''
    select and return indexes of top N probabilities in outputs array
    
    parameters
    ----------
        outputs - array-like prediction output from ANN
        N - number of top probability entries to be returned
        
    returns
    -------
        array of N indexes of N highest probabilities in outputs array
        
    '''
    if isinstance(outputs,np.ndarray):
        return outputs.argsort()[-N:][::-1]
    
    return outputs.detach().cpu().numpy().argsort()[-N:][::-1]


def get_accuracy(outputs,labels,device):
    '''
    calculate accuracy
    
    outputs - ANN prediction, tensor like
        1st dim runs along samples
        2nd dim runs along classes
        
    labels - GT vector of class labels (converted to integers)
    
    It finds the index of the maximal probability prediction from outputs and compares
    it with labels
    
    returns:
    
    number of correct predictions divided by the size of the intput vector
    
    '''

    # acc2=(outputs==labels).cpu().float().squeeze().detach().numpy()
    acc2=nn.MSELoss().to(device)
    return acc2(outputs,labels).cpu().float().detach().numpy()

    # print(acc2)
#


    return acc2

class vec_dist_norm2():
    def __init__(self):
        '''
        '''
        
    def calc(self,output,target):
        '''
        output, target - 3d tensors or shape N x 1 x data_size
    
        returns 1d tensor of size N 
        '''
        d=torch.sqrt(torch.sum((output-target)**2,dim=-1))/torch.std(target,dim=-1)
        # self.append(d)
        # return self
        return d.detach().numpy()
    
    
class TrainHistory():
    def __init__(self, name='train history'):
        '''
        '''
        self.train_hist={'epoch' : [], 'loss':[]}
        self.name=name

    def load(self,history_file):
        '''
        load train history from txt file consistent with
        epoch loss acc
        '''
        d=np.loadtxt(os.path.join(history_file))
        for i,k in enumerate(self.train_hist.keys()):
            self.train_hist[k]=list(d[:,i])


    def save(self,fname):
        '''
        '''
        tmp=np.array(self.train_hist['epoch'])
        
#         with open(fname) as f:
        for k in self.train_hist.keys():
#             print(k)
            if k!='epoch':
#                 if isinstance(self.train_hist[k], (list,np.array,tuple)):
                tmp=np.vstack([tmp,self.train_hist[k]])
        
        
        np.savetxt(fname,tmp.T,header=' '.join(self.train_hist.keys()))
        
    def dict2df(self):
        '''
        convert dictionary to pandas dataframe
        
        Each key will become pandas column.
        Assumes that keys contain lists the same length
        '''
        
    def last(self, key):
        return self.train_hist[key][-1]


class Train_History_EpochLossAcc(TrainHistory):
    def __init__(self, name='train history'):
        super().__init__(name)
        self.train_hist={'epoch' : [], 'loss':[], 'acc': []}
        

    def append(self,epoch,loss,acc):
        self.train_hist['epoch'].append(epoch)
        self.train_hist['loss'].append(loss)
        self.train_hist['acc'].append(acc)
    
    def load(self, history_file):
        '''
        load train history from txt file consistent with
        epoch loss acc
        '''
        d=np.loadtxt(os.path.join(history_file))
        self.train_hist['epoch']=list(d[:,0])
        self.train_hist['loss']=list(d[:,1])
        self.train_hist['acc']=list(d[:,2])

      
    def last(self, key):
        return self.train_hist[key][-1]



