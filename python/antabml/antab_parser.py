'''
Created on Dec 9, 2021

@author: blew
'''
import os,sys
import random,binascii

__version__ = 0.1
__date__ = '2021-06-22'
__updated__ = '2021-06-22'

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

def get_parser():
    
    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1] if len(__import__('__main__').__doc__.split("\n"))>=2 else ''
    program_epilog ='''
    
Examples:

python ./train_antab.py --bs 1024 --model autoenc --model_dir ../../models/autoenc  --train_dir ../../data/train/
 
 
python ./train_antab.py --bs 1024 --epoch 10000 --model dense --model_dir ../../models/denseFF/ --train_dir ../../data/train/ --load_worker 5

run logged with MLflow
python ./train_antab.py --bs 1000 --epoch 500 --model dense --dsize 2000 --denseConf 1000 500 500 1000 --dropout 0.1 --loss smoothL1 --lr 0.01  --train_dir ~/programy2/antab-vlbeer-data/wisdom-tr/ --split 0.5 0.5 0 --model_dir ~/programy2/antabML/models/auto --load_worker 4 --MLflow_tracking_uri "http://192.168.1.63:5000" --MLflow_run_name "dense 2000 1 DS=tr"
'''
    program_license = '''%s

  Created by Bartosz Lew on %s.
  Copyright 2021 Bartosz Lew. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, epilog=program_epilog, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]", default=0)
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        parser.add_argument('--train_dir', type=str, default='../../data/train/',help='train dir')
        # parser.add_argument(dest="paths", help="path to csv file with data to plot [default: %(default)s]", metavar="path", nargs='*')
        # parser.add_argument('-o','--outfile', type=str, default='',help='output file prefix')
        parser.add_argument('--split', nargs='+', type=float,
                            help='train dataset split [default: %(default)s]', 
                            default=[0.6,0.2,0.2])
        parser.add_argument('--split_seed', type=int,
                            help='random seed for torch.Generator.manual_seed() [default: %(default)s]',default=1)
        parser.add_argument('--lr', type=float, 
                            help='optimizer learning rate[default: %(default)s]', 
                            default=0.001)
        parser.add_argument('--momentum', type=float, 
                            help='optimizer momentum [default: %(default)s]', 
                            default=0.9)
        parser.add_argument('--dropout', type=float, 
                            help='dropout rate after linear layer [default: %(default)s]', 
                            default=0.1)
        parser.add_argument('--device', type=str, 
                            help='select device to use [default: %(default)s]', 
                            choices=['auto','cpu','cuda:0'], 
                            default='auto')

        parser.add_argument('--file_ext', type=str,
                            help='input file extension [default: %(default)s]', 
                            default=['awpkl'],
                            nargs='*',
                            )
        parser.add_argument('--model_dir', type=str, 
                            help=''''directory containing trained model and all 
                            partial files. The default indicates the root dir
                            for all models, and magic key 'auto' indicates that 
                            the model directory will be automatically generated
                            with which the root dir will be suffixed. [default: %(default)s]''', 
                            required=False,
                            default='../../models/auto')

        parser.add_argument('--chkpt_save', type=int, 
                            help='save ckp file every this epoch [default: %(default)s]', 
                            required=False,
                            default=50)

        parser.add_argument('--model', type=str, 
                            help='model name [default: %(default)s]', 
                            required=False,
                            default='class',
                            choices=['class','autoenc','lstm','dense','conv1d'])
        parser.add_argument('--denseConf',type=int,
                            help='dense model linear hidden layers sizes configuration',
                            default=[],
                            nargs='*')
        parser.add_argument('--loss', type=str, 
                            help='loss function name [default: %(default)s]', 
                            required=False,
                            default='smoothL1',
                            choices=['smoothL1','L1','MSE'])

        parser.add_argument('--epochs', type=int, help='Number of epochs during training [default: %(default)s]', default=100)
        parser.add_argument('--bs', type=int, help='batch size [default: %(default)s]', default=10000)
        parser.add_argument('--load_workers', type=int, help='batch load workers number [default: %(default)s]', default=1)

        parser.add_argument('--dsize',type=int,default=1000,help='''Data size. Incompatible data
        will be padded with zeros on the input and output or truncated
        ''')
        # parser.add_argument('--plot_img_count', action='store_true',
        #                     help='plot image count per day and cumulatively',
        #                     default=False)

        parser.add_argument('--MLflow_tracking_uri',type=str,default='',
                            help='''MLflow tracking server e.g. "http://192.168.1.63:5000.
                            Use empty string to skip MLflow logging''')
        parser.add_argument('--MLflow_exp_name',type=str,default='antabML',help='MLflow experiment name. If empty will generate name using datetime')
        parser.add_argument('--MLflow_run_name',type=str,default='',help='MLflow run name.')

        '''
        inference options
        '''
        parser.add_argument('--model_file', type=str, 
                            help='path to trained model. Used for inference [default: %(default)s]', 
                            required=False,
                            default='../../models/model.pth')

        parser.add_argument('--test_file', type=str, 
                            help='path to awpkl file [default: %(default)s]. Eg. "../../data/train/blew-May-Jun21-ec077dtr.01.awpkl"', 
                            required=False,
                            default='')
        


        # parser.add_argument('--title',type=str,default='', help='''
        # Plot title
        # ''')

        # Process arguments
        args = parser.parse_args()
        
        if args.model_dir.endswith('auto'):
            # suffix=''.join([str(random.randint(0, i)) for i in range(20)])
            suffix=binascii.hexlify(os.urandom(8)).decode()
            args.model_dir=os.path.join(args.model_dir[:-4],suffix)
            
        # print(args.model_dir)

    except KeyboardInterrupt:
        ## handle keyboard interrupt ###
        return 0
#     except Exception as e:
#         if DEBUG or TESTRUN:
#             raise(e)
#         indent = len(program_name) * " "
#         sys.stderr.write(program_name + ": " + repr(e) + "\n")
#         sys.stderr.write(indent + "  for help use --help")
#         return 2
        
    return args
