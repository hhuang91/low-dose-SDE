# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:56:27 2023

@author: hhuang91
"""
import numpy as np
from torch_radon import RadonFanbeam
from _structs import gStruct

def getRFP(g,imSize = 256, imRes = 1.):
    scaFact = 1./imRes
    if isinstance(g, gStruct):
        angles = g.angle * np.pi / 180
        SAD = g.SAD
        DAD = g.SDD - g.SAD
        detCount = g.XYdim[0]
        detSize = g.PixSize[0]
    else:
        angles = g['angle'][0][0][0] * np.pi / 180
        SAD = g['SAD'][0][0][0][0]
        DAD = g['SDD'][0][0][0][0] - g['SAD'][0][0][0][0]
        detCount = g['UVWdim'][0][0][0][0]
        detSize = g['PixSize'][0][0][0][0]
    # scale up geometery as a way to increase resolution
    SAD *= scaFact
    DAD *= scaFact
    detSize  *= scaFact
    return RadonFanbeam(imSize, angles, SAD, DAD, detCount, detSize ,clip_to_circle=False)
    