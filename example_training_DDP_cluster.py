# -*- coding: utf-8 -*-
"""
Example of training on computing clusters
with DDP (data distributed parallel)

@author: hhuang91
"""
import sys
sys.path.append("/home/hhuang91/Libs/Git/LDSDE")
from _trainNetworkDDP import Train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int)
parser.add_argument('--ngpu', type=int)
args = parser.parse_args()

deviceN = 'cuda'
gLoc = './g.mat'
lr = 1e-5
wMSE = 0
wFbpMSE = 1

Train(  args,
        deviceN,
        gLoc,
        lr,
        wMSE,wFbpMSE,
        continueTrain=False,xferLearn=True,
        EPOCHS=1000,batchSize=25,
        dataPath='./cleanPrj.h5',outDir = './networkOutput/LDSDE_100steps_FBPxfer',
        )