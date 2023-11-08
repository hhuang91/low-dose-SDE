# -*- coding: utf-8 -*-
"""
Wrapper function for creating a differentiable radon fan beam projector
based on torch-radon [https://github.com/matteo-ronchetti/torch-radon]

@author: hhuang91
"""
import numpy as np
from torch_radon import RadonFanbeam
from _structs import gStruct

def getRFP(g:gStruct,imSize:int = 256, imRes:float = 1.) -> RadonFanbeam:
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
    