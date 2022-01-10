'''
Created on Dec 9, 2021

@author: blew
'''
import os,sys
import numpy as np
import loaders,utils,stats,ann_models
import torch,torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torch import Generator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow


class antab_trainer():
    '''
    classdocs
    '''


    def __init__(self, args, **kwargs):
        '''
        Constructor
        '''
        self.args=args
        self.logfile='train.log'
        self.logger=utils.get_logging('train-model', os.path.join(args.model_dir,self.logfile))

        self.dsize=args.dsize


        self.cuda_device_count=torch.cuda.device_count()
        self.deviceName="cpu"
        self.device=args.device
#         self.deviceName=args.device
        if args.device=='auto':
            self.device = torch.device(self.deviceName)
            if self.cuda_device_count>0:
                self.deviceName="cuda:0"
                self.device = torch.device(self.deviceName)
            torch.device(self.device)
        elif args.device=='cpu':
            self.deviceName="cpu"
            self.device='cpu'
        elif args.device=='cuda':
            self.deviceName="cuda"
            self.device='cuda'
        else:
            self.deviceName=args.device
            self.device=args.device

        if args.verbose>2:
            self.logger.info("Using devide: {}".format(self.device))


        #
        # MLflow stuff
        #
        self.MLflow=False
        self.MLflowRun=None
        if args.MLflow_tracking_uri!='':
            self.MLflow=True
            mlflow.set_tracking_uri(args.MLflow_tracking_uri)
            mlflow.set_experiment(args.MLflow_exp_name)
#             if args.MLflow_run_name!='':
            self.MLflowRun=mlflow.start_run(run_name=args.MLflow_run_name)

        
    ##############################################################
    def process_model_epoch(self, data_loader, ann, optimizer, criterion, 
                            train, args, 
#                             trainConfig=None,
                            **kwargs):
        '''
        train ann a single epoch using data from data_loader
        
        
            
        returns
        -------
            tuple(loss, stats)
            
            where:
                loss is a list of losses in all processed data and 
                stats is a dictionary containing various statistics of the model. 
        '''
        
        # set default kwargs
        test=False
        epoch=None
        if 'epoch' in kwargs:
            epoch=kwargs['epoch']
        else:
            epoch=-1
            
        if 'test' in kwargs:
            test=kwargs['test']
        
        
        running_loss = []
        estats={ 'loss' : running_loss, 
                'acc' : []
                }
        
        # m=nn.ConstantPad1d((0,))
        nbatch=len(data_loader)
        for i, (inputs, targets) in enumerate(data_loader, 1):
            if args.verbose>2:
                print('inputs.shpe: ',inputs.shape)
                print('targets.shpe: ',targets.shape)
            # x=inputs.view(len(inputs),1,-1)
            # y=inputs.view(len(targets),1,-1)
            x=inputs.view(len(inputs),-1)
            y=targets.view(len(targets),-1)
            bstats=self.process_batch(i, nbatch,x, y, 
                                     ann, optimizer, criterion, train, args,**kwargs)
        
        
        estats['loss'].append(bstats['loss'])
        estats['acc'].append(bstats['acc'])
            

        '''
        save MLflow stuff
        '''
        MLFmetric_type_pref='train'
        if train:
            MLFmetric_type_pref='train'
        elif test==False:
            MLFmetric_type_pref='valid'
        else:
            MLFmetric_type_pref='test'

            
        if args.MLflow_tracking_uri!='':
            mlflow.log_metric('{} mean accuracy'.format(MLFmetric_type_pref),avg_stats['acc'],epoch)
            mlflow.log_metric('{} mean loss'.format(MLFmetric_type_pref),np.mean(bstats['loss']),epoch)
            mlflow.log_metric('epoch',float(epoch))

        
        return {'loss': np.mean(estats['loss']), 'acc': np.mean(estats['acc'])}

    
    ##############################################################
    def process_batch(self,i, nbatch, 
                      inputs, targets, ann, optimizer, criterion, 
                      train, args, **kwargs):
        '''
        parameters
        ==========
            i - batch number (log only)
            i2 - minibatch number (for log only)
            nbatch - total number of batches
            
        '''
        # set default kwargs
        test=False
        epoch=''
        if 'test' in kwargs:
            test=kwargs['test']
        if 'epoch' in kwargs:
            epoch=kwargs['epoch']

        if train:
            optimizer.zero_grad()

        # forward + backward + optimize
        outputs,_ = ann(inputs.to(self.device))

        if self.args.verbose>2:
            print('inputs',inputs.shape)
            print('outputs',outputs.shape)
            print('targets',targets.shape)
        
        loss = criterion(outputs, targets.to(self.device))
        if self.args.verbose>4:
            print('inputs')
            print(inputs)
            print('targets')
            print(targets)
            print('outputs')
            print(outputs)

#             print(loss)
        if train:
            loss.backward()
            optimizer.step()
            
        bloss=loss.item()
        # bacc=np.mean(stats.vec_dist_norm2().calc(outputs,targets))
        bacc=np.mean(stats.get_accuracy(outputs,targets))


        if self.args.verbose>1:
            self.logger.info("epoch: {}, batch: {}/{}, mean loss: {}, mean acc: {}".format(
                epoch,i,nbatch,
                bloss,bacc
                ))
        return {'loss': bloss, 'acc' : bacc}

    
    def train_model(self):
        '''
        '''
        args=self.args
        logger=self.logger
        
        logger.info("Training starts")
        
        trainConfig={} # holds DS and network and training setup information
        
        DS=loaders.antab_loader(root=args.train_dir, 
                                dsize=args.dsize,
                                transform=None, 
                                target_transform=None,
                                verbose=args.verbose)

        Ntrain,Nvalid,Ntest=[int(frac*len(DS)) for frac in args.split]
        if (Ntrain+Nvalid+Ntest!=len(DS)):
            Ntest=len(DS)-Ntrain-Nvalid
        logger.info("Size of train dataset: {}".format(Ntrain))
        logger.info("Size of valid dataset: {}".format(Nvalid))
        logger.info("Size of test dataset: {}".format(Ntest))

        trainset, validset, testset=random_split(DS,[Ntrain,Nvalid,Ntest],
                                                 generator=torch.manual_seed(args.split_seed))


        trainloader=DataLoader(trainset,
                            batch_size=args.bs,
                            num_workers=args.load_workers)
    
        validloader=DataLoader(validset,
                                batch_size=args.bs,
#                                 shuffle=trainConfig['shuffle'],
                                num_workers=args.load_workers)
    
        testloader=DataLoader(testset,
                                batch_size=args.bs,
#                                 shuffle=trainConfig['shuffle'],
                                num_workers=args.load_workers)

        trainConfig['DSsize']=len(DS)
        trainConfig['DSsplit']=args.split
        trainConfig['split_seed']=args.split_seed
        trainConfig['Ntrain']=Ntrain
        trainConfig['Nvalid']=Nvalid
        trainConfig['Ntest']=Ntest
        trainConfig['bs']=args.bs
        trainConfig['epochs']=args.epochs

        lossfn=None
        if args.model=='class':
            net= ann_models.DenseFF([args.dsize, args.dsize,args.dsize,args.dsize,args.dsize//2,args.dsize//4,1]) 
            lossfn= nn.BCELoss()
        elif args.model=='lstm':
            net= nn.LSTM(args.dsize, args.dsize, batch_first=True) 
            lossfn= nn.NLLLoss()
        elif args.model=='autoenc':
            # net= ann_models.DenseFF([args.dsize, args.dsize//2,args.dsize//4,args.dsize//4,args.dsize//2,args.dsize]) 
            # net= ann_models.DenseFF([args.dsize, args.dsize*2,args.dsize*4,args.dsize*4,args.dsize*2,args.dsize,1]) 
            # net= ann_models.DenseFF([args.dsize, args.dsize//2,args.dsize//4,args.dsize//8,args.dsize//4,args.dsize//2,args.dsize]) 
            net= ann_models.DenseFF([args.dsize, args.dsize//8,args.dsize//16,args.dsize//8,args.dsize]) 
            net= ann_models.DenseFF([args.dsize, args.dsize,args.dsize,args.dsize,args.dsize], nntype='autoenc') 
            lossfn= nn.MSELoss()

        Nparam,Names=utils.ANN_parameter_count(net, trainable_only=True)
        for name, param in zip(Names,Nparam):
            logger.info("layer name: {}, params count: {}".format(name,param))
        trainConfig['model_trainable_Nparams']=int(np.sum(Nparam))
        logger.info("Total trainable parms count: {}".format(trainConfig['model_trainable_Nparams']))
        
        # lossfn= nn.CrossEntropyLoss()
        
        trainConfig['optimizer']='SDG'
        trainConfig['optimizer_lr']=args.lr
        trainConfig['optimizer_momentum']=args.momentum
        optimizer = optim.SGD(net.parameters(), 
                              lr=trainConfig['optimizer_lr'], 
                              momentum=trainConfig['optimizer_momentum'])
        
        

        train_hist=stats.Train_History_EpochLossAcc('Train DS')
        valid_hist=stats.Train_History_EpochLossAcc('Valid DS')

        # '''
        # save MLflow stuff
        # '''
        # if args.MLflow_tracking_uri!='':
        #     [ mlflow.log_param(k,v) for k,v in trainConfig.items() ]
        #     [ mlflow.log_param(k,v) for k,v in netConfig.items() ]

        start_epoch=0
        # if args.load_ckpt!="":
        #     start_epoch=self.state['epoch']
            # logger.info('Loading last trained epoch: {}'.format(start_epoch))
#             start_epoch+=1
        epoch=start_epoch


        for epoch in range(start_epoch+1,args.epochs+1):  # loop over the dataset multiple times
        
            '''
            train on train dataset
            '''
            logger.info('training')
            train_stats=self.process_model_epoch(data_loader=trainloader, 
                                                  ann=net, 
                                                  optimizer=optimizer, 
                                                  criterion=lossfn, 
                                                  train=True, 
                                                  args=args,
                                                  epoch=epoch
#                                                       trainConfig=trainConfig,
                                                  )
            
            train_hist.append(epoch,
                              train_stats['loss'],
                              train_stats['acc'],
                              )
    
#             if epoch % args.chkpt_save == 0:
#                 # save checkpoint
#                 chkpoint_file=os.path.join(args.model_dir,"model.ckp_%i" % epoch)
#                 logger.info("Saving check point to file: {}".format(chkpoint_file))
#                 torch.save({
#                         'epoch': epoch,
#                         'model_state_dict': net.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
# #                         'loss': train_hist.last('loss')
#                         'loss': lossfn.state_dict(),
#                         'acc' : train_stats,
#                         }, 
#                         chkpoint_file
#                 )
                
    
    
            
            '''
            test on validation dataset
            '''
            logger.info('validating')
            valid_stats=self.process_model_epoch(data_loader=validloader, 
                                                  ann=net, 
                                                  optimizer=optimizer, 
                                                  criterion=lossfn, 
                                                  train=False, 
#                                                       trainConfig=trainConfig,
                                                  args=args,
                                                  epoch=epoch,
                                                  )
#             print(valid_loss,valid_stats)
            
            valid_hist.append(epoch,
                              valid_stats['loss'],
                              valid_stats['acc'],
                              )
    
            '''
            info
            '''
            logger.info("epoch: {}, train/valid acc: {:.3f}/{:.3f}, loss: {:.3f}/{:.3f}".format(
                epoch,
                train_stats['acc'],valid_stats['acc'],
                train_stats['loss'],valid_stats['loss'],
            ))

        '''
        Save model at epoch end
        '''
        model_file_name=os.path.join(args.model_dir,"model.pth")
        logger.info("Saving model to: {}".format(model_file_name))
        torch.save(net.state_dict(),model_file_name)

        hist_file_name=os.path.join(args.model_dir,"train_hist.txt")
        train_hist.save(hist_file_name)
        hist_file_name=os.path.join(args.model_dir,"valid_hist.txt")
        valid_hist.save(hist_file_name)











