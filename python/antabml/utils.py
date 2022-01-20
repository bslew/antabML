'''
Created on Dec 9, 2021

@author: blew
'''

import logging
import os,sys
import datetime
import numpy as np
import errno
import stats
import numpy as np
import matplotlib.pyplot as plt

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise  


def get_logging(name,fname,mode='a', lvl=logging.INFO):
    r'''
    Create and return a new logger to be used.
    
    Creates a logger object with name `name` that will log to file `fname`.
    The log file can be appended depending on the `mode` parameter.
    
    Parameters
    ----------

    name : str
        Logger name
        
    fname : str
        Log file name
        None to disable logging to file
        
    mode : str {'a', 'w'}, optional
        Log file open mode
        
    lvl : ``log level as defined in logging module`` {logging.DEBUG}
        
    Returns
    -------
    logger 
        logger object
    
    Raises
    ------
    
    Notes
    -----
    Notes here
    
    References
    ----------
    refs
    
    Examples
    --------
    >>> logger=ACCloggers.get_logging('train-model', 'logfile', 'a')
    
    '''
    print("Starting logger {} with level {} (log file: {})".format(name,lvl,fname))
    if fname!=None:
        if os.path.dirname(fname)!='':
            os.makedirs(os.path.dirname(fname),exist_ok=True)
    
    # create logger with 'spam_application'
    logger = logging.getLogger(name)
    logger.setLevel(lvl)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")

    
    # create file handler which logs even debug messages
    if fname!=None:
        fh = logging.FileHandler(fname,mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(ch)

    
    return logger



def ANN_parameter_count(net, trainable_only=True) -> list:
    '''
    Calculate number of parameters in a network
    
    Parameters
    ----------
        net - torch.nn subbclass
        
        trainable - if True then calculates only the parameters for
            which gradient derivative is calculated
    
    Returns
    -------
        Returns a dublet: a list containing
        number of parameters in each layer 
        (weights and biases are calculated separately)
        and a list containing layer names
    '''
    
    Nparam=[]
    Names=[]
    if trainable_only:
        for name, param in net.named_parameters():
            if param.requires_grad:
                Nparam.append(np.prod(param.data.size()))
                Names.append(name)
    else:
        for name, param in net.named_parameters():
            Nparam.append(np.prod(param.data.size()))
            Names.append(name)
        
    return Nparam,Names



def plot_train_history(train_hist, output_file=None, show=False):
    '''
    plot train history object
    
    train_hist - list of train_history objects that contain train_hist dictionary attributes
    compatible with
    
    train_hist={'epoch' : [], 'loss':[], 'acc': []}
    name - name to be plotted as label
    
    This function is dedicated for plotting training history of 
    classification networks.
    
    '''
    
    if not show:
        import matplotlib
        matplotlib.use('Agg')
    
    DScolors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#     from cycler import cycler
#     print(matplotlib.rcParams['axes.prop_cycle'].color)
#      = cycler(color='bgrcmyk')
    
    fig=plt.figure(figsize=(12,8))
    plt.subplot(211)
    for i,th in enumerate(train_hist):
        lab=None
        if isinstance(th, stats.Train_History_EpochLossAcc):
            if i==0:
                lab='top-1'
            plt.plot(th.train_hist['epoch'],th.train_hist['acc'],  label=lab, 
                     lw=2, ls='-', c=DScolors[i])
            plt.ylabel('accuracy')
            
        elif isinstance(th, stats.Train_History_EpochLossAcc135):
            if i==0:
                lab='top-1'
            plt.plot(th.train_hist['epoch'],th.train_hist['acc1'], label=lab, 
                     lw=2, ls='-', c=DScolors[i])
            if i==0:
                lab='top-3'
            plt.plot(th.train_hist['epoch'],th.train_hist['acc3'], label=lab, 
                     lw=1, ls='--', c=DScolors[i])
            if i==0:
                lab='top-5'
            plt.plot(th.train_hist['epoch'],th.train_hist['acc5'], label=lab, 
                     lw=1, ls='-.', c=DScolors[i])
            
            plt.ylabel('accuracy')
            plt.legend(ncol=3)
            
#             plt.legend(ncol=3)
        plt.ylabel('measure')
    plt.grid(True)
        

    plt.subplot(212)
    for i,th in enumerate(train_hist):
        plt.plot(th.train_hist['epoch'],th.train_hist['loss'], label=th.name)    

    plt.grid(True)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    plt.legend()

    if output_file is not None:
        plt.savefig(output_file)

    if show:
        plt.show()
    
