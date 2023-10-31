# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:00:43 2023
Data loader for LPSDE training
@author: hhuang91
"""

# %% Import
import h5py
import torch
# import torch.nn as nn
# import numpy as np
from _LDSDE import LDSDE


# %% Dataloader wrapper
class hdf5Loader():
    def __init__(self,datPath,batchSize,useCuda,numWorker=0):
        loaderKwagrs = {'batch_size':batchSize,'shuffle':True}
        if useCuda:
            loaderKwagrs.update({'num_workers': numWorker,'pin_memory': True})
        self.datPath = datPath
        self.trainDS = hdf5DataSet(datPath,'Train')
        self.valdnDS = hdf5DataSet(datPath,'Valdn')
        self.testDS = hdf5DataSet(datPath,'Test')
        self.trainLoader = torch.utils.data.DataLoader(self.trainDS,**loaderKwagrs)
        self.valdnLoader = torch.utils.data.DataLoader(self.valdnDS,**loaderKwagrs)
        self.testLoader  = torch.utils.data.DataLoader(self.testDS, **loaderKwagrs)
    def getLoader(self):
        return self.trainLoader,self.valdnLoader,self.testLoader
    
    
# %% Dataloader Wrapper for DDP
class hdf5LoaderDDP():
    def __init__(self,datPath,batchSize,useCuda,args,numWorker=0):
        loaderKwagrs = {'batch_size':batchSize,'shuffle':False}
        if useCuda:
            loaderKwagrs.update({'num_workers': numWorker,'pin_memory': True})
        self.trainDS = hdf5DataSet(datPath,'Train')
        self.valdnDS = hdf5DataSet(datPath,'Valdn')
        self.testDS = hdf5DataSet(datPath,'Test')
        trainSampler = torch.utils.data.distributed.DistributedSampler(
                    self.trainDS,
                    num_replicas=args.ngpu,
                    rank=args.local_rank,
                    shuffle=True
        )
        valdnSampler = torch.utils.data.distributed.DistributedSampler(
                    self.valdnDS,
                    num_replicas=args.ngpu,
                    rank=args.local_rank,
                    shuffle=True
        )
        testSampler = torch.utils.data.distributed.DistributedSampler(
                    self.testDS,
                    num_replicas=args.ngpu,
                    rank=args.local_rank,
                    shuffle=True
        )
        self.trainLoader = torch.utils.data.DataLoader(self.trainDS,sampler=trainSampler,**loaderKwagrs)
        self.valdnLoader = torch.utils.data.DataLoader(self.valdnDS,sampler=valdnSampler,**loaderKwagrs)
        self.testLoader  = torch.utils.data.DataLoader(self.testDS, sampler=testSampler, **loaderKwagrs)
    def getLoader(self):
        return self.trainLoader,self.valdnLoader,self.testLoader    
    

# %% Dataset Objects
class hdf5DataSet(torch.utils.data.Dataset):
    def __init__(self,datPath,datKind):
        self.datPath = datPath
        self.datKind  = datKind
        self.SDE = LDSDE()
    def __getitem__(self, idx):
        with h5py.File(self.datPath,'r') as f:
            # step 1: get clean projection
            prjDS = f[self.datKind+'Prj']
            x_neg1 = prjDS[idx,:,:].squeeze()
            # step 2: sample a random T as prior and t0 resevse steps for training 
            # to ease the training, we assume that x_T always has the lowest dose
            step = self.SDE.N - 1
            T = self.SDE.discrete_t[step]
            # step 3: sample training data
            x_t, alpha, t = self.SDE.sampleTrainingData(x_neg1, T)
            return x_neg1, x_t, alpha, t
    def __len__(self):
        with h5py.File(self.datPath,'r') as f:
            tmpDS= f[self.datKind+'Prj']
            length = tmpDS.shape[0]
        return length