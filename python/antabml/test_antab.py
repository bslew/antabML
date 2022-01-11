#!/usr/bin/env python
'''

@author:     Bartosz Lew

@contact:    bartosz.lew@protonmail.com
@deffield    updated: Updated
'''

import sys
import os

import utils
import trainer
import antab_parser
import matplotlib.pyplot as plt


__all__ = []

DEBUG = 0
TESTRUN = 0
PROFILE = 0

def plot_test(res):
    plt.plot(res['input'], 'ok', ls='-' ,label='input')
    plt.plot(res['target'], 'og', label='target')
    plt.plot(res['output'], 'sr', label='output')
    plt.legend()
    plt.show()
    


def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    args = antab_parser.get_parser()
    # paths = args.paths
    verbose = args.verbose

    if verbose > 0:
        print("Verbose mode on")
    
    if not os.path.isfile(args.model_dir):
        utils.mkdir_p(args.model_dir)
        
    tr = trainer.antab_trainer(args)
    tr.load_trained_model()
    res=tr.test()
    print(res)
    plot_test(res)

    return 0

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
        sys.argv.append("-r")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'utils.plotting.acc_plot_atlas_cumulative_image_count_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())
    