# -*- coding: utf-8 -*-
"""
FBP loss function
Using differentibale backprojector (https://github.com/matteo-ronchetti/torch-radon)
to reconstruction image first
then compute loss on image domain 
(so graident will be propagated back to projection domain)

@author: hhuang91
"""

#%%
import torch
from _projector import getRFP 
from _structs import gStruct
from _filterTorch import fbpFilterRaisedCosine2D_MB
import torch.nn as nn
#%% Directly compute loss on FBP-ed image
class fbpMSE():
    def __init__(self,g:gStruct,recDim:int = 256,recVoxSize:float = 0.25):
        self.g = g
        self.L = nn.MSELoss()
        self.RFP = getRFP(g, recDim, recVoxSize)
    def __call__(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        xF = torch.zeros_like(x)
        yF = torch.zeros_like(y)
        ## Input has dimension (batch, channel = 1, pixel, andgle)
        ## The filter function need (bacth, pixel, 1, angle)
        ## and RFP input needs (batch, channel = 1, pixel, angle )
        xF = fbpFilterRaisedCosine2D_MB(x.squeeze(1).unsqueeze(2), 1, 0.5, self.g, norm = [1], parker = 0, extrap = 1).squeeze(2).unsqueeze(1)
        yF = fbpFilterRaisedCosine2D_MB(y.squeeze(1).unsqueeze(2), 1, 0.5, self.g, norm = [1], parker = 0, extrap = 1).squeeze(2).unsqueeze(1)
        ## xF and yF has dimension (Batch, Channel, Pixel, Angle)
        ## But torch radon wants (Batch, Channel, Angle, Pixel)
        uX = self.RFP.backprojection(xF.transpose(2,3))
        uY = self.RFP.backprojection(yF.transpose(2,3))
        loss = self.L(self.norm(uX),self.norm(uY))
        return loss
    def norm(self,x):
        # deprecated, becuase normalization will mess up aboslute value of noise
        return x