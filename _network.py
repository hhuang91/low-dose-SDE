# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:10:54 2023
Cross Domian denoising network
@author: hhuang91
"""

#%% Import
import torch
import torch.nn as nn
import torch.nn.functional as F
#%% normalization method
normMethod =nn.InstanceNorm2d
#%% uNet builder helper funciton
def uNetLayer(inChannel, outChannel):
    return nn.Sequential(
                            nn.Conv2d(inChannel, outChannel, 3, padding = 1),
                            normMethod(outChannel),
                            nn.LeakyReLU(),
                            nn.Conv2d(outChannel, outChannel, 3, padding = 1),
                            normMethod(outChannel),
                            nn.LeakyReLU()
                            )
def upConvolve(inChannel, outChannel):
    return nn.Conv2d(inChannel,outChannel,1)

def splitPrj(x):
    thrsld1 = 0.55
    thrsld2 = 0.85
    x1 = torch.clamp(x,None,thrsld1)
    x2 = torch.clamp(x,thrsld1,thrsld2)
    x3 = torch.clamp(x,thrsld2,None)
    res = torch.cat((x1,x2,x3),1)
    return res
#%% cross(X)-Domain Noise Reduction
class XDNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.timeEncode = nn.Sequential(
                                        nn.Linear(1,8),
                                        nn.Linear(8, 64),
                                        nn.Linear(64, 8),
                                        nn.Linear(8, 1))
        self.xtEncode = nn.Sequential(nn.Conv2d(3,24,3,groups=3,padding=1),
                                        nn.Conv2d(24,32,3,padding=1),
                                        nn.Conv2d(32,16,3,padding=1),
                                        nn.Conv2d(16,1,1))
        self.xTEncode = nn.Sequential(nn.Conv2d(3,24,3,groups=3,padding=1),
                                        nn.Conv2d(24,32,3,padding=1),
                                        nn.Conv2d(32,16,3,padding=1),
                                        nn.Conv2d(16,1,1))
        cNumBase = 16 #32
        self.downSample = nn.MaxPool2d(3)
        self.downLayer1 = uNetLayer(3, cNumBase)
        self.downLayer2 = uNetLayer(cNumBase, cNumBase*2)
        self.downLayer3 = uNetLayer(cNumBase*2, cNumBase*4)
        self.downLayer4 = uNetLayer(cNumBase*4, cNumBase*8)
        self.downLayer5 = uNetLayer(cNumBase*8, cNumBase*16)
        self.upCovn54 = upConvolve(cNumBase*16,cNumBase*8)
        self.upCovn43 = upConvolve(cNumBase*8,cNumBase*4)
        self.upCovn32 = upConvolve(cNumBase*4,cNumBase*2)
        self.upCovn21 = upConvolve(cNumBase*2,cNumBase)
        self.upLayer4 = uNetLayer(cNumBase*16, cNumBase*8)
        self.upLayer3 = uNetLayer(cNumBase*8, cNumBase*4)
        self.upLayer2 = uNetLayer(cNumBase*4, cNumBase*2)
        self.upLayer1 = uNetLayer(cNumBase*2, cNumBase)
        self.finalConv = nn.Conv2d(cNumBase, 1, 1)
    def forward(self, x_TIN, x_tIN, tIN):
        x_T = self.xTEncode(splitPrj(x_TIN))
        x_t = self.xtEncode(splitPrj(x_tIN))
        tE = self.timeEncode(torch.clamp(tIN,0,1))*torch.ones_like(x_t)
        x = torch.cat((x_T,x_t,tE),1)
        downOut1 = self.downLayer1(x)
        downOut2 = self.downLayer2(
                                    self.downSample(downOut1)
                                    )
        downOut3 = self.downLayer3(
                                    self.downSample(downOut2)
                                    )
        downOut4 = self.downLayer4(
                                    self.downSample(downOut3)
                                    )
        downOut5 = self.downLayer5(
                                    self.downSample(downOut4)
                                    )
        upOut5 = downOut5
        upOut4 = self.upLayer4(
                                torch.cat(
                                            (F.upsample_bilinear(
                                                                    self.upCovn54(upOut5),downOut4.shape[-2:]
                                                                )
                                             , downOut4
                                             )
                                            ,1
                                          )
                                )
        upOut3 = self.upLayer3(
                                torch.cat(
                                            (F.upsample_bilinear(
                                                                    self.upCovn43(upOut4),downOut3.shape[-2:]
                                                                )
                                             , downOut3
                                             )
                                            ,1
                                          )
                                )
        upOut2 = self.upLayer2(
                                torch.cat(
                                            (F.upsample_bilinear(
                                                                    self.upCovn32(upOut3),downOut2.shape[-2:]
                                                                )
                                             , downOut2
                                             )
                                            ,1
                                          )
                                )
        upOut1 = self.upLayer1(
                                torch.cat(
                                            (F.upsample_bilinear(
                                                                    self.upCovn21(upOut2),downOut1.shape[-2:]
                                                                )
                                             , downOut1
                                             )
                                            ,1
                                          )
                                )
        out = self.finalConv(upOut1)
        return out

#%% helper function for loading the network states
def loadState(net,stateFN):
    state = torch.load(stateFN,next(net.parameters()).device)
    cnnState = state['cnnState']
    try:
        net.load_state_dict(cnnState)
    except:
        cnnState = stateDictConvert(cnnState)
        net.load_state_dict(cnnState)
    print('loaded training state')

def stateDictConvert(DPstate):
    from collections import OrderedDict
    State = OrderedDict()
    for k, v in DPstate.items():
        name = k.replace("module.", "") # remove 'module.' of dataparallel
        State[name] = v
    return State