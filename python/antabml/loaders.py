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
from torch.nn import ConstantPad1d, ReflectionPad1d

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
    loads wisdom saved by BL modified antabfs program
    
    
    keywords
    --------
        len - requested length of output arrays. Default full array.
            If specified random part of the array is requested length is returned
            
    returns
    -------
        1d array-like x,1d array-like y,status :bool
    '''
    with open(path, 'rb') as f:
        data=pickle.load(f,encoding="latin1")
    
    if 'X' not in data.keys():
        return [],[],False
    if 'Y' not in data.keys():
        return [],[],False
    x,y=np.array(data['X']),np.array(data['Y'])
    # print("Loading {}/{} data points from {}".format(len(x),len(y),path))
    if len(x)!=len(y):
        return x,y,False
        # raise Exception("BadData")
                   
    if len(x)<=1:
        return x,y,False
    
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
            
            preload - True/False - preload all data before training
    
            model - 'class', 'dense' (default)
        
        Example
        -------
        
        
        
        '''
        super().__init__()

            
        self.X=None
        self.Y=None
        
        self.run_mode='train'
        if 'run_mode' in kwargs.keys():
            self.run_mode=kwargs['run_mode']
        
        self.verbose=0
        if 'verbose' in kwargs.keys():
            self.verbose=kwargs['verbose']

        self.model='dense'
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
                if 'preload' in kwargs.keys():
                    if kwargs['preload']:
                        self.preload_all_data()
                
            elif self.run_mode=='inference':
                self.files_list=ListSubdirectory(
                    root_dir=self.root).getRecursiveFileList(self.file_filter)
                self.files_list=[os.path.join(self.root,x) for x in self.files_list]
            elif self.run_mode=='test':
                pass
        print('self.run_mode ',self.run_mode)
        print('self.root ',self.root)
        

        self.logger=utils.get_logging('loader', "loader.log")
        if self.verbose>1:
            self.logger.info('loader mode: {}'.format(self.model))
#         print('mapping: ',self.mapping)
            
    def preload_all_data(self):
        '''
        '''
        print('preloading all data')
        self.X=None
        self.Y=None
        Xl=[]
        Yl=[]
        N=0
        for i,f in enumerate(self.files_list):
            f=self.get_path(i)
            X,Y,n=self.load_as_batch(f)
            # self.data=torch.stack(self.data)
            
            for x,y in zip(X,Y):
                N=N+len(X)
                Xl.append(x)
                Yl.append(y)
                # self.X=X if self.X==None else torch.vstack((self.X,x))
                # self.Y=Y if self.Y==None else torch.vstack((self.Y,y))
                
            # if len(self.X) % 100==0:
            #     print('loaded {} vectors'.format(len(self.X)))
            if len(Xl) % 100==0 or self.verbose>2:
                print('loaded {} vectors'.format(len(Xl)))

        self.X=torch.Tensor(len(Xl),self.dsize)
        self.Y=torch.Tensor(len(Yl),self.dsize)
        
        for i,(x,y) in enumerate(zip(Xl,Yl)):
            self.X[i]=x
            self.Y[i]=y
                
        
        
        if self.model=='class':
            X=self.X.numpy()
            Y=self.Y.numpy()
            self.Y=torch.tensor( np.array(np.abs((Y-X)/(X+np.spacing(1)))>0.01,dtype=int) ).float()
            # print(self.Y[0])
            # print('preloading for class done')
            # sys.exit(0)

        # print(self.X.shape)

        # self.data=torch.stack(self.data,dim=0)
        print('preloading done')
        
    def get_path(self,index):
        if self.run_mode=='train':
            file_path=os.path.join(self.root,self.files_list[index])
        elif self.run_mode=='inference':
            file_path=self.files_list[index]
        else:
            Exception('unknown run mode')

        return file_path

    def pad_data(self,x,**kwargs):
        '''
        returns padded version of x 

        parameters
        ----------
            x - 1d tensor


        returns
        -------
            2-d tensor with padded values of size N x self.dsize

        keywords
        --------
            mode 
                - replicate - replicates the last value in the inputs and targets to fill the required
                fixed data vector length that matches the size of first/last ANN linear layer

                - reflection
                
                
        
        '''
        padx=x
        if 'mode' in kwargs.keys():
            if kwargs['mode']=='replicate':
                self.pad_value=x[-1]
                
        if 'mode' in kwargs.keys():
            if kwargs['mode']=='reflection':
                N=len(x) // self.dsize
                Npad=(N+1)*self.dsize - len(x) 
                padx=x.reshape((1,-1))
                # print('len(x):',len(x))
                # print(padx.shape[1])
                # print(Npad)
                padTimes=0
                while padx.shape[1]<=Npad:
                    padTimes+=1
                    # print('need to pad couple  times ({})'.format(padTimes))
                    thispad=padx.shape[1]-1
                    # print(thispad)
                    pad = ReflectionPad1d((0, thispad))
                    padx=pad(padx)
                    Npad=(padx.shape[1] // self.dsize +1)*self.dsize - padx.shape[1] 
                #     print(Npad)
                #     print(padx.shape)
                # print('finally')
                # print(padx.shape[1])
                Npad=(padx.shape[1] // self.dsize +1)*self.dsize - padx.shape[1] 
                pad = ReflectionPad1d((0, Npad))
                padx=pad(padx)
                # print(Npad)
                # print(padx.shape)

        else:
            N=len(x) // self.dsize
            ps=(N+1)*self.dsize - len(x) 
            pad=ConstantPad1d((0,ps),self.pad_value)
            padx=pad(x)
        # print(len(pad(x)),ps,len(x),self.dsize)

        return padx.view(-1,self.dsize)

    def load_as_batch(self,path):
        '''
        load wisdom train data from pickle file and form a batch
        of vectors of length self.dsize
        
        parameters
        ----------
        path - path to picke file
        
        returns
        -------
        tuple x,y,n where x,y are 2-d tensors of size Nxdsize with self.pad_value
        right-padding (if required) and n is the length of the original data
        
        n=-1 if load error occured (e.g. corrupted/incomplete input file)
        
        '''
        if self.verbose>2:
            print('loading wisdom from {}'.format(path))
        x,y,status=load_train_wisdom(path=path)
        if not status:
            return torch.Tensor(),torch.Tensor(),-1
        n=len(x)
        x=torch.from_numpy(x).float()
        y=torch.from_numpy(y).float()

        # x=self.pad_data(x,mode='replicate')
        # y=self.pad_data(y,mode='replicate')
        x=self.pad_data(x,mode='reflection')
        y=self.pad_data(y,mode='reflection')
        if self.verbose>3:
            print('padded X',x.shape)
        return x,y,n
        
    def __getitem__(self, index):
        '''
        load and preprocess input image
        
        returns
        -------
        
            in train mode: tuple of img,gt,(root,gt_path)
            in inference mode: tuple of img,root
        
        '''

        if self.X!=None:
            return self.X[index].float(),self.Y[index].float()
                
        file_path=self.get_path(index)

        status=False
        # if self.model=='conv1d':
        #     x,y,status=load_train_wisdom(file_path)
        #     file_path=self.get_path(random.randint(0,len(self.files_list)-1))
        #     return torch.from_numpy(x).float(), torch.from_numpy(y).float()

        while not status:
            x,y,n=self.load_as_batch(path=file_path)
            status=n>0
            file_path=self.get_path(random.randint(0,len(self.files_list)-1))

        if self.model=='class':
            # return x.float(), torch.tensor(y[0]==x[0]).float()
            return x.float(), torch.tensor( np.array(np.abs((y-x)/(x+np.spacing(1)))>0.01,dtype=int) ).float()
        elif self.model=='autoenc':
            return x.float(), x.float()
        elif self.model=='dense':
            return x.float(), y.float()
        elif self.model=='conv1d':
            return x.float(), y.float()
        
        # while not status:
        #     # print('Trying to load data from file: {}'.format(file_path))
        #     x,y,status=load_train_wisdom(file_path, len=self.dsize)
        #     file_path=self.get_path(random.randint(0,len(self.files_list)-1))
        #
        # # print(x)
        # # print(torch.from_numpy(y==x).float())
        #
        # # return torch.from_numpy(x).float(), torch.from_numpy(y==x).float()
        # # print(torch.from_numpy(x).float(), torch.from_numpy(y==x).int())
        # # return torch.from_numpy(x).float(), torch.from_numpy(
        # #     np.array([int(y[0]==x[0]),int(y[0]!=x[0])],dtype=float)).float()
        # if self.model=='class':
        #     return torch.from_numpy(x).float(), torch.tensor(y[0]==x[0]).float()
        # elif self.model=='autoenc':
        #     return torch.from_numpy(x).float(), torch.from_numpy(x).float()
        # elif self.model=='dense':
        #     return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        # elif self.model=='conv1d':
        #     return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            
        return None
        
            
    
    def __len__(self):
        if self.X!=None:
            return len(self.X)
        
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
