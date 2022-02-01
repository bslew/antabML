'''
Created on Dec 9, 2021

@author: blew
'''
import unittest
import loaders

import matplotlib.pyplot as plt

class Test(unittest.TestCase):


    def testLoadPicke(self):
        data=loaders.load_wisdom('../../data/train/blew-May-Jun21-es098btr.04.awpkl')
    
        print(data.keys())
        print('x: ',data['x'].shape)
        print('x0: ',len(data['x0']))
        print('y: ',data['y'].shape)
        print('y0: ',len(data['y0']))
    
    
        plt.plot(data['x'],data['y0'], label='y0')
        plt.plot(data['x'],data['y'], label='y')
        plt.legend()
        plt.show()
    #
    # def test_train_wisdom(self):
    #     x,y=loaders.load_train_wisdom('../../data/train/blew-May-Jun21-es098btr.04.awpkl')
    #     print('x: ',x.shape)
    #     print('y: ',y.shape)
    #
    # def test_train_wisdom_maxlen(self):
    #     print('maxlen load test')
    #     x,y=loaders.load_train_wisdom('../../data/train/blew-May-Jun21-es098btr.04.awpkl', maxlen=1000)
    #     print('x: ',x.shape)
    #     print('y: ',y.shape)

    def testLoader(self):
        DS=loaders.antab_loader(root='../../data/train', 
                                dsize=100,
                                transform=None, 
                                target_transform=None)

        for x in DS:
            print(x)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLoadPicke']
    unittest.main()