'''
Created on Dec 9, 2021

@author: blew
'''
import os,sys
import yaml, pickle
import numpy as np
import utils
# import torch
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import ConstantPad1d

def load_wisdom(path):
    '''
    loads wisdom saved by BL modified antab program
    
    '''
    data=None
    with open(path, 'rb') as f:
        data=pickle.load(f,encoding="latin1")
    return data

def load_train_wisdom(path, **kwargs):
    '''
    loads wisdom saved by BL modified antab program
    
    returns x and y as np arrays
    
    keywords
    --------
        len - requested length of output arrays. Default full array.
            If specified random part of the array is requested length is returned
            
    returns
    -------
        (x,y),status :bool
    '''
    with open(path, 'rb') as f:
        data=pickle.load(f,encoding="latin1")
    
    if 'y0' not in data.keys():
        return [],[],False
    if 'y' not in data.keys():
        return [],[],False
    x,y=np.array(data['y0']),np.array(data['y'])
    # print("Loading {}/{} data points from {}".format(len(x),len(y),path))
    if len(x)!=len(y):
        return x,y,False
        # raise Exception("BadData")
                   
    
    if 'len' in kwargs.keys() and int(kwargs['len'])>0:
        dlen=int(kwargs['len'])
        if dlen>len(x):
            return x,y,False
            s="kwargs['len'] ({}) should be < len(x) ({})".format(dlen,len(x))
            raise Exception(s)
        # print(dlen)
        st=random.randint(0,len(x)-dlen)
        st=0
        # print('st',st)
        x=x[st:st+dlen]
        y=y[st:st+dlen]
    return x,y,True



class antab_loader(Dataset):
    '''
    This is dataset loader for generated antabs
    
    '''
    def __init__(self, root='', file_filter=('awpkl'),
                 transform=None,
                 target_transform=None,
                 dsize=-1,
                 pad_value=0,
                 **kwargs
                 ):
        '''
        
        Parameters
        -----------
        
            dsize - size of returned featue. Incompatible input data will be padded with 
                pad values
            pad_value - padding values
            
            
        Keywords
        --------
        
            files - list of file name strings. If present, the files from the list will be processed
                from root/img_dir_name/[files] and the parameters target_transform and target_dir_name
                are ignored.
        
        
            run_mode - ('train', 'inference'), default: 'train'
            
        
        Example
        -------
        
        
        
        '''
        super().__init__()
        
        self.run_mode='train'
        if 'run_mode' in kwargs.keys():
            self.run_mode=kwargs['run_mode']
        
        self.verbose=0
        if 'verbose' in kwargs.keys():
            self.verbose=kwargs['verbose']

        self.model='autoenc'
        if 'model' in kwargs.keys():
            self.model=kwargs['model']
        
        self.root=root
        self.file_filter=file_filter
        self.pad_value=pad_value
        self.dsize=dsize
      
        self.files_list=[]
        if 'files' in kwargs.keys():
            self.run_mode='inference'
            self.files_list=kwargs['files']
        else:
            if self.run_mode=='train':
                self.files_list=ListSubdirectory(
                    root_dir=self.root).getRecursiveFileList(self.file_filter)
            elif self.run_mode=='inference':
                self.files_list=ListSubdirectory(
                    root_dir=self.root).getRecursiveFileList(self.file_filter)
                self.files_list=[os.path.join(self.root,x) for x in self.files_list]
                
        print('self.run_mode ',self.run_mode)
        print('self.root ',self.root)
        

        self.logger=utils.get_logging('loader', "loader.log")
        if self.verbose>1:
            self.logger.info('loader mode: {}'.format(self.model))
#         print('mapping: ',self.mapping)
            
    def get_path(self,index):
        if self.run_mode=='train':
            file_path=os.path.join(self.root,self.files_list[index])
        elif self.run_mode=='inference':
            file_path=self.files_list[index]
        else:
            Exception('unknown run mode')

        return file_path

    def reshape_data(self,x):
        '''
        
        '''
        ps=x % self.dsize
        pad=ConstantPad1D((0,ps),self.pad_value)
        
        
    def __getitem__(self, index):
        '''
        load and preprocess input image
        
        returns
        -------
        
            in train mode: tuple of img,gt,(root,gt_path)
            in inference mode: tuple of img,root
        
        '''
        file_path=self.get_path(index)

        status=False
        # if self.model=='conv1d':
        #     x,y,status=load_train_wisdom(file_path)
        #     file_path=self.get_path(random.randint(0,len(self.files_list)-1))
        #     return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        
        while not status:
            # print('Trying to load data from file: {}'.format(file_path))
            x,y,status=load_train_wisdom(file_path, len=self.dsize)
            file_path=self.get_path(random.randint(0,len(self.files_list)-1))
                
        # print(x)
        # print(torch.from_numpy(y==x).float())
        
        # return torch.from_numpy(x).float(), torch.from_numpy(y==x).float()
        # print(torch.from_numpy(x).float(), torch.from_numpy(y==x).int())
        # return torch.from_numpy(x).float(), torch.from_numpy(
        #     np.array([int(y[0]==x[0]),int(y[0]!=x[0])],dtype=float)).float()
        if self.model=='class':
            return torch.from_numpy(x).float(), torch.tensor(y[0]==x[0]).float()
        elif self.model=='autoenc':
            return torch.from_numpy(x).float(), torch.from_numpy(x).float()
        elif self.model=='dense':
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        elif self.model=='conv1d':
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            
        return None
        
            
    
    def __len__(self):
        return len(self.files_list)        






class ListSubdirectory():
    '''
    Directory content lister.
    This is not a generator, do not use for big data
    '''
    def __init__(self,root_dir):
#         self.file_type=file_type
        self.root_dir=root_dir
        
    def getFileList(self,ftype=''):
        '''
        Return a depth-1 list of directory 
        files with matching extension.
        
        The returned list items do not contain the root_dir prefix
        '''
        if not os.path.isdir(self.root_dir):
            return None

        return [f for f in os.listdir(self.root_dir)
                    if os.path.isfile(os.path.join(self.root_dir,f)) and 
                    f.endswith(ftype)]
        
    def getDirectoryList(self,select=''):
        '''
        Return a depth-1 list of directory 
        directories with matching selection criteria
        
        The returned list items do not contain the root_dir prefix
        '''
        return [f for f in os.listdir(self.root_dir)
                    if os.path.isdir(os.path.join(self.root_dir,f)) and 
                    f.endswith(select)]
        
    def getRecursiveFileList(self,ftype=''):
        '''
        Traverse root_dir and build and return a list of matching files
        
        The returned list items do not contain the root_dir prefix.

        
        parameters
        ----------
        
            ftype - string (e.g 'jpg') or tuple e.g. ('jpg','png') or
                    list ['jpg','png']
                    
        examples
        --------
        
        if the directory structure is:
        
        dir1/
            dir11/
                file1.jpg
                file2.jpg
            
        dir2/

        >>> Loaders.ListSubdirectory(root_dir='dir1').getRecursiveFileList('jpg')
        
        ['dir11/file1.jpg',dir11/file2.jpg']
        
        '''
#         print(ftype)
#         import sys
#         sys.exit()
        if isinstance(ftype, list):
            ftype=tuple(ftype)
        
        fl=[]
        cdir=os.getcwd()
        os.chdir(self.root_dir)
        for root, dirs, files in os.walk('.'):
#             print(root,dirs,files)
            for file in files:
                if file.endswith(ftype):
                    fl.append(os.path.join(root,file)[2:])
                    
        os.chdir(cdir)
        return fl
