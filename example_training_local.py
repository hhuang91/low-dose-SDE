# -*- coding: utf-8 -*-
"""
Example to train LDSDE on local (windows) PC

@author: hhuang91
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append(r'F:\f_users\heyuan\Git\LDSDE_scoreMatching')
from _trainNetwork import Train

#%% start training
deviceN = 'cuda:1'
gLoc = './g.mat'
lr = 1e-3
wMSE = 1
wFbpMSE = 0
Train(deviceN,
          gLoc,
          lr,
          wMSE,wFbpMSE,
          continueTrain=False,xferLearn=True,
          EPOCHS=500,batchSize=1,
          dataPath='./cleanPrj.h5',outDir = './networkOutput/LDSDE_Testing/',
          numWorker=0,
          dispOn=True)