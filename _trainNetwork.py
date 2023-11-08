# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:48:58 2023

Wrapper and training class the LDSDE network

@author: hhuang91

"""
#%% Import

from __future__ import print_function
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import scipy.io
import os
import sys
import matplotlib.pyplot as plt
from _dataLoader import hdf5Loader
from _network import XDNR
from _structs import gStruct
from _LDSDE import LDSDE
import random

#%% Wrapper Function:   
def Train(deviceN:str,
          gLoc:str,
          lr:float,
          wMSE:float, wFbpMSE:float,
          continueTrain:bool = False,xferLearn:bool = False,
          prjSize:List[int] = [1024,360],recDim:int = 512,recVoxSize:float = 0.25,
          EPOCHS:int = 500,batchSize:int = 8,
          dataPath:str = './Data',outDir:str = './networkOutput/',
          numWorker:int = 0,
          dispOn:bool = False):
    # set device
    device = torch.device(deviceN)
    if device.type =='cuda' and device.index == None:
        cnn = nn.DataParallel(XDNR().to(device));
    else:
        cnn = XDNR().to(device)
    # get geometry
    tmp = scipy.io.loadmat(gLoc)
    gRaw = tmp['g']
    g = gStruct()
    g.setFromSioLoaded(gRaw)
    # set optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    # set dataloader
    dataLoader = hdf5Loader(dataPath, batchSize, device.type =='cuda',numWorker)
    # construct training class
    tNt = trainNTest(
                        device,
                        g,
                        cnn,
                        optimizer,wMSE,wFbpMSE,
                        continueTrain,xferLearn,
                        prjSize,recDim,recVoxSize,
                        EPOCHS,
                        dataLoader,outDir,
                        dispOn
                        )
    # TRAIN!!!!
    tNt.train()
    
def Test(deviceN:str,
          gLoc:str,
          lr:float,
          wMSE:float, wFbpMSE:float,
          testAtEpoch: int,
          rndSeed:int = 0,
          continueTrain:bool = False,xferLearn:bool = False,
          prjSize:List[int] = [1024,360],recDim:int = 512,recVoxSize:float = 0.25,
          EPOCHS:int = 500,batchSize:int = 8,
          dataPath:str = './Data',outDir:str = './networkOutput/',
          numWorker:int = 0,
          dispOn:int = False):
    # set device
    device = torch.device(deviceN)
    if device.type =='cuda' and device.index == None:
        cnn = nn.DataParallel(XDNR().to(device));
    else:
        cnn = XDNR().to(device)
    # get geometry
    tmp = scipy.io.loadmat(gLoc)
    gRaw = tmp['g']
    g = gStruct()
    g.setFromSioLoaded(gRaw)
    # set optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    # set dataloader
    dataLoader = hdf5Loader(dataPath, batchSize, device.type =='cuda',numWorker)
    # construct training class
    tNt = trainNTest(
                        device,
                        g,
                        cnn,
                        optimizer,wMSE,wFbpMSE,
                        continueTrain,xferLearn,
                        prjSize,recDim,recVoxSize,
                        EPOCHS,
                        dataLoader,outDir,
                        dispOn
                        )
    # Test!!!!
    tNt.test(testAtEpoch,rndSeed)

#%% Training and testing class
class trainNTest():
    def __init__(self,
                 device:str,
                 g:gStruct,
                 cnn:torch.nn.modules,
                 optimizer:torch.optim,
                 wMSE:float, wFbpMSE:float,
                 continueTrain:bool ,xferLearn:bool,
                 prjSize:List[int], recDim:int ,recVoxSize:float,
                 EPOCHS:int,
                 dataLoader:hdf5Loader,
                 outDir:str,
                 dispOn:bool =False,
                 rank:int = 0):
        self.rank = rank
        self.g = g
        self.lossFncMSE = torch.nn.MSELoss()
        if wFbpMSE > 0:
            from _lossFnc import fbpMSE
            self.lossFncFbpMSE = fbpMSE(g,recDim,recVoxSize)
        else:
            self.lossFncFbpMSE = lambda x,y: torch.tensor(0.)
        self.cnn = cnn;
        self.optimizer = optimizer;
        self.wMSE = wMSE
        self.wFbpMSE = wFbpMSE
        self.outDir = outDir;
        self.dispOn = dispOn;
        for param_group in optimizer.param_groups:
            self.lr = param_group['lr']
        self.prjSize = prjSize;
        self.device = device;
        self.EPOCHS = EPOCHS
        self.trainLoader,self.valdnLoader,self.testLoader = dataLoader.getLoader()
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        self.trainLoss = np.array([])
        self.valdnLoss = np.array([])
        self.lossDataFN = self.outDir+'/'+"lossData_LR_"+str(self.lr)+".mat"
        self.stateN = self.outDir+'/'+"state_lr_"+str(self.lr)+"_Epoch_"
        self.startEpoch = 0;
        self.SDE = LDSDE(device = device)
        if xferLearn:
            print("Transfer Learning")
            self.lossDataFN = self.outDir+'/'+"lossDataXfer_LR_"+str(self.lr)+".mat"
            self.stateN = self.outDir+'/'+"stateXfer_lr_"+str(self.lr)+"_Epoch_"
            if not continueTrain:
                stateFN = self.outDir+'/' + "stateXfer"+".pth.tar"
                if os.path.exists(stateFN):
                    self.loadState(stateFN,ldOptm=False,partialLoad=True)
                    print(f"State loaded:{stateFN}")
                else:
                    raise Exception(f"Need state file named {stateFN}")
        if continueTrain:
            print("Continue Training")
            if os.path.exists(self.lossDataFN):
                self.trainLoss = scipy.io.loadmat(self.lossDataFN)["TrainLoss"].squeeze()
                self.valdnLoss = scipy.io.loadmat(self.lossDataFN)["ValdnLoss"].squeeze()
                print(f"Loss data loaded:{self.lossDataFN}")
            else:
                raise Exception("Loss Data file missing!")
            self.startEpoch = max(len(self.trainLoss),len(self.valdnLoss));
            stateFN = self.stateN + str(self.startEpoch-1)+".pth.tar";
            if os.path.exists(stateFN):
                self.loadState(stateFN)
                print(f"State loaded:{stateFN}")
            else:
                raise Exception("State file missing!")
                
    def train(self):
        print("Begin Training")
        print(f"Start from {self.startEpoch} Epoch")
        for epoch in range(self.startEpoch,self.EPOCHS):
            trainLoss = 0
            self.cnn.train()
            datIter = tqdm(self.trainLoader,file=sys.stdout,desc="Training")
            for batch_idx, data in enumerate(datIter):
                loss, fbpLoss, mseLoss = self.reverseLoop(data, datIter, train = True)
                datIter.set_description(f"Training, batch loss: {loss}, fbp Loss: {fbpLoss}, mse Loss: {mseLoss} ")
                trainLoss += loss
                if self.rank == 0:
                    state = {'cnnState' : self.cnn.state_dict(),
                             'optimizerState' : self.optimizer.state_dict()}
                    stateFN = self.stateN+str(epoch)+"_.pth.tar";
                    self.saveState(state,stateFN)
            trainLoss /= len(self.trainLoader)
            valLoss = self.valdn()
            print(f"Epoch: {epoch}. Training Loss: {trainLoss}. Validation Loss: {valLoss}")
            self.trainLoss = np.append(self.trainLoss,trainLoss)
            self.valdnLoss = np.append(self.valdnLoss,valLoss)
            if self.rank == 0:
                scipy.io.savemat(self.lossDataFN, {'TrainLoss':self.trainLoss,
                                                      'ValdnLoss':self.valdnLoss})
                state = {'cnnState' : self.cnn.state_dict(),
                          'optimizerState' : self.optimizer.state_dict()}
                stateFN = self.stateN+str(epoch)+".pth.tar";
                self.saveState(state,stateFN)
                
    @torch.no_grad()
    def valdn(self,valdnAtEpoch = -1):
        if valdnAtEpoch >= 0:
                stateFN = self.stateN+str(valdnAtEpoch)+".pth.tar"
                if os.path.exists(stateFN):
                    self.loadState(stateFN,ldOptm = False,partialLoad=True)
                    print(f'Previous state loaded @{valdnAtEpoch} Epoch for validation')
                else:
                    raise Exception(f'State @{valdnAtEpoch} Epoch is not found.')
        print("Begin Validation")
        valdnLoss = 0
        datIter = tqdm(self.valdnLoader,file=sys.stdout,desc="Validating")
        self.cnn.eval()
        for batch_idx, data in enumerate(datIter):
            loss, fbpLoss, mseLoss = self.reverseLoop(data, datIter, train = False)
            valdnLoss += loss
        valdnLoss /= len(self.valdnLoader)
        return valdnLoss
    
    @torch.no_grad()
    def test(self,testAtEpoch = -1,rndSeed = 0):
        testLoss = 0
        if testAtEpoch >=0:
            stateFN = self.stateN+str(testAtEpoch)+".pth.tar"
            if os.path.exists(stateFN):
                self.loadState(stateFN,ldOptm = False,partialLoad=True)
                print(f'Previous state loaded @{testAtEpoch} Epoch for testing')
            else:
                raise Exception(f'State @{testAtEpoch} Epoch is not found.')
        print("Begin Testing.")
        self.cnn.eval()
        torch.manual_seed(rndSeed)
        np.random.seed(rndSeed)
        random.seed(rndSeed)
        datIter = tqdm(self.testLoader, file=sys.stdout,desc="Testing")
        for batch_idx, data in enumerate(datIter):
            loss, fbpLoss, mseLoss = self.reverseLoop(data, datIter, train = False)
            testLoss += loss
        testLoss /= len(self.testLoader)
        testResults = {'testLoss':testLoss}
        saveFN = self.outDir+'/'+"test_lr_"+str(self.lr)+'_Epoch_'+str(testAtEpoch)
        scipy.io.savemat(saveFN+'.mat', testResults)
        
    def reverseLoop(self,data,pBar,train = False):
        revAllStepLoss = 0.
        revAllStepMSEloss = 0.
        revAllStepFBPloss = 0.
        x_neg1, x_T = self.reshapeImg(data[0:2])
        alphaT, t = self.reshapeSca(data[2:])
        x_T_Norm = x_T/alphaT
        x_t = x_T.clone()
        dt = self.SDE.rev_dt
        while (t>0).any():
            t , alpha, D = self.SDE.discretizedValues_at_T(t)
            noise = self.SDE.getNoiseGT(x_neg1,x_t,alpha)
            noise_est = self.cnn(x_T/alphaT,x_t/alpha,t)
            lossInput = (x_neg1, x_t,alpha, noise,noise_est)
            if train:
                self.optimizer.zero_grad()
                loss, fbpLoss, mseLoss = self.getLoss(*lossInput)
                loss.backward()
                self.optimizer.step()
            else:
                loss, fbpLoss, mseLoss = self.getLoss(*lossInput)
            revAllStepLoss += loss.item()
            revAllStepMSEloss += mseLoss.item()
            revAllStepFBPloss += fbpLoss.item()
            with torch.no_grad():
                dx = self.SDE.getReverseSDE_dx(x_t, t, x_T_Norm, noise)
            x_t = x_t + dx
            t = t + dt
            pBar.set_description(f"Reversing...t = {t[0].item()}")
        if self.dispOn:
            self.plot(x_t,noise_est,noise)
        return revAllStepLoss/1000, revAllStepFBPloss/1000, revAllStepMSEloss/1000
    
    def getLoss(self,x_neg1, x_t, alpha, noise, noise_est):
        mseLoss = self.lossFncMSE( noise_est,  noise )*self.wMSE
        x_neg1_est = self.SDE.noiseEst2Xneg1(x_t ,noise_est, alpha, x_neg1) #if self.wFbpMSE > 0 else 0.
        fbpLoss = self.lossFncFbpMSE(x_neg1,x_neg1_est)*self.wFbpMSE
        loss = (mseLoss + fbpLoss)
        return loss, fbpLoss, mseLoss
    
    @torch.no_grad()
    def plot(self,x_t,noise_est,noise):
        plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(x_t[0,0,:,:].detach().cpu().numpy(),cmap='gray');plt.colorbar()
        plt.subplot(3,1,2)
        plt.imshow((noise_est[0,0,:,:]).detach().cpu().numpy(),cmap='gray');plt.colorbar()
        plt.subplot(3,1,3)
        plt.imshow((noise[0,0,:,:]).detach().cpu().numpy(),cmap='gray');plt.colorbar()
        plt.show()
        
    def reshapeImg(self,data:tuple):
        res = [x.to(self.device).view(-1,1,*x.shape[-2:]).float() for x in data]
        if len(data) > 1 :
            return tuple(res)
        else:
            return tuple(res)[0]
        
    def reshapeSca(self,data:tuple):
        res = [x.to(self.device).view(-1,1,1,1).float() for x in data]
        if len(data) > 1 :
            return tuple(res)
        else:
            return tuple(res)[0]
        
    def saveState(self,state,stateFN,disp=True):
        torch.save(state,stateFN)
        if disp:
            print('-->current training state saved')
            
    def loadState(self,stateFN,ldOptm = True,partialLoad = False):
        state = torch.load(stateFN,self.device)
        cnnState = state['cnnState']
        try:
            self.cnn.load_state_dict(cnnState)
        except:
            cnnState = self.stateDictConvert(cnnState)
        finally:
            if partialLoad:
                cnnState = self.partialStateConvert(cnnState)
            self.cnn.load_state_dict(cnnState)
        self.cnn.load_state_dict(cnnState)
        print('loaded training state')
        if ldOptm:
            self.optimizer.load_state_dict(state['optimizerState'])
            print('loaded optimizer state')
            
    def stateDictConvert(self,DPstate):
        from collections import OrderedDict
        State = OrderedDict()
        for k, v in DPstate.items():
            name = k.replace("module.", "") # remove 'module.' of dataparallel
            State[name] = v
        return State
    
    def partialStateConvert(self,DPstate):
        cnnDict = self.cnn.state_dict()
        partialState = {k: v for k, v in DPstate.items() if k in cnnDict}
        cnnDict.update(partialState)
        return cnnDict
