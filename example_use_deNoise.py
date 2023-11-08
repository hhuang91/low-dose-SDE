# -*- coding: utf-8 -*-
"""
Example Code to use LDSDE
@author: hhuang91
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import scipy.io as sio
import torch
import numpy as np
sys.path.append(r'F:\f_users\heyuan\Git\LDSDE_scoreMatching')
from deNoise import doseIncreaserGT, doseIncreaser
from _LDSDE import LDSDE
#%%
tmp = sio.loadmat('./Test/2DbCT_Test.mat')
prjs = tmp['prjs']
g = tmp['g']
x_neg1 = torch.tensor(prjs[0,:,:].squeeze()).view(1,1,1024,360)
#%% get prior and target
testT = 1
SDE = LDSDE(device = 'cuda:1')
x_t,_ = SDE.sampleXt(x_neg1,testT)
x_0,_ = SDE.sampleXt(x_neg1,0)
#%% network output
deNoiseObj = doseIncreaser(stateFN = './trainedState.pth.tar',device = 'cuda:1')
deNoiseObj.cnn.train()
x_dN = deNoiseObj.removeNoise(x_t, testT, nIter = 20)
x_hD = deNoiseObj.increaseDose(x_t, testT, plot = False)
sio.savemat('LDSDE_test_netOut_fbp_avg_trainMode.mat', {'prjN': x_t.squeeze().cpu().numpy(), 
                           'prjN_hD':x_hD.squeeze().cpu().numpy(),
                           'prjN_dN': x_dN.squeeze().cpu().numpy(),
                            'prjN_targ': x_0.squeeze().cpu().numpy(),
                            'prjClean':x_neg1.squeeze().cpu().numpy()})

#%% GT test
deNoiseObj = doseIncreaserGT('cuda:1')
x_dN = deNoiseObj.doseIncreaserGTtest(x_t, testT, x_neg1, plot=True)
sio.savemat('LDSDE_test_noiseMatchingTest.mat', {'prjN': x_t.squeeze().cpu().numpy(), 
                            'prjN_dN': x_dN.squeeze().cpu().numpy(),
                            'prjN_targ': x_0.squeeze().cpu().numpy()})
