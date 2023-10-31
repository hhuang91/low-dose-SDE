# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:04:29 2023

@author: hhuang91
"""

#%%
import torch
from _projector import getRFP 
from _filterTorch import fbpFilterRaisedCosine2D_MB
import torch.nn as nn
#%% Directly compute loss on FBP-ed image
class fbpMSE():
    def __init__(self,g,recDim = 256,recVoxSize = 0.25):
        self.g = g
        self.L = nn.MSELoss()
        self.RFP = getRFP(g, recDim, recVoxSize)
    def __call__(self, x, y):
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
        # x = x.clone() - x.min()
        # x = x.clone() / x.max()
        return x