# -*- coding: utf-8 -*-
"""
Pytorch version of FBP filter
Allowing for multi-bacth filtering and autograd

@author: hhuang91
"""

import torch as tr
import numpy as np
from tqdm import tqdm

#%% Scatter correction
def scatter_corr(p, spr):
    pfy = pyyf(p)
    if pfy == 'lineInt':
        y = np.exp(-p)
    else:
        y = p
    for ijk in tqdm(range(y.shape[2]),desc = 'SPR Antiscattering'):
        tmp = y[:,:,ijk]
        scat = (spr/(1+spr))*np.percentile(tmp,spr)
        y[:,:,ijk] = tmp - scat
    y[y<0] = 0
    if pfy == 'lineInt':
        return -np.log(y)
    else:
        return y
    
#%% FBP Filter    
def fbpFilterRaisedCosine2D(y, fact, hamming, g, showPrg = False, **kwargs):

    (nu, nv, nb) = y.shape
    device = y.device

    if 'parker' in kwargs:
        parkerOn = kwargs['parker']
    else:
        parkerOn = 0

    if 'extrap' in kwargs:
        extrapOn = kwargs['extrap']
    else:
        extrapOn = 0

    if 'vFilter' in kwargs:
        vFilterOn = kwargs['vFilter']
    else:
        vFilterOn = 0

    if 'fact_v' in kwargs:
        fact_v = kwargs['fact_v']
    else:
        fact_v = 0.8

    if 'hamming_v' in kwargs:
        hamming_v = kwargs['hamming_v']
    else:
        hamming_v = 0.5

    if 'norm' in kwargs:
        norm = kwargs['norm']
    else:
        pyf = pyyf(y)
        if pyf == 'lineInt':#'p':
            norm =[]
            print('Note: Assuming input is Line Integral (log corrected)')
        elif pyf == 'prj':
            norm =[1]
            print('Note: Assuming input is projection (no Log correction)')
        #norm = []
    
    if g.SDD.size == 1:#len(g.SDD) == 1:
        DS = np.tile(g.SDD, nb)
        A = np.tile(g.SAD, nb)
        u0 = np.tile(g.u0, nb)
        v0 = np.tile(g.v0, nb)
    else:
        DS = g.SDD
        A = g.SAD
        u0 = g.u0
        v0 = g.v0
    
    # Pixel sizes in mm
    uPixelSize = g.PixSize[0]
    vPixelSize = g.PixSize[1]

    # Filter specification in u
    NU = int(2**np.ceil(np.log2(nu)))
    filter = np.zeros((2*NU,1))
    arg1 = np.pi * fact * np.arange(0,nu) # s coordinates where the filter is calculated
    h1 = hamming * vfunc(arg1) + 0.5 * (1-hamming) * (vfunc(np.pi + arg1) + vfunc(np.pi - arg1))
    arg2 = np.pi * fact * np.arange(-nu+1, 0)
    h2 = hamming * vfunc(arg2) + 0.5 * (1-hamming) * (vfunc(np.pi + arg2) + vfunc(np.pi - arg2))
    filter[0:nu] = np.expand_dims(h1, axis=1)
    filter[(2*NU-nu+1):(2*NU)] = np.expand_dims(h2, axis=1)
    filter = 2 * (fact**2) / (2*uPixelSize*2) * filter
    filter = np.fft.fft(filter, axis=0)
    filter = tr.tensor(filter).to(device)

    # Filter specification in v (if needed)
    if vFilterOn:
        NV = int(2**np.ceil(np.log2(nv)))
        vFilter = np.zeros([2*NV,1])
        arg1 = np.arange(0, NV)
        h1 = hamming_v + (1-hamming_v) * np.cos(np.pi / (fact_v * NV) * arg1)
        if (np.ceil(fact_v*NV)+1)<len(h1):
            h1[np.ceil(fact_v*NV)+1:] = 0
        arg2 = np.arange(-NV,0)
        h2 = hamming_v + (1-hamming_v) * np.cos(np.pi / (fact_v * NV) * np.abs(arg2))
        if len(h2) > np.ceil(fact_v*NV):
            h2[0:(len(h2) - np.ceil(fact_v*NV) - 1)] = 0
        vFilter[0:NV] = np.expand_dims(h1, axis=1)
        vFilter[NV:] = np.expand_dims(h2, axis=1)

    if parkerOn:
        wPker = applyParker(nu, nv, nb, u0, DS, uPixelSize, g.angle)

    y_filt = tr.zeros((nu,nv,nb),dtype=tr.float).to(device)
    epsilon = tr.tensor(1e-10).to(device)

    if showPrg:
        iterItem = tqdm(np.arange(0, nb),desc = 'Filtering')
    else:
        iterItem = np.arange(0, nb)

    for b in iterItem:

        p_mat = uPixelSize * np.matmul(np.ones((nu,1),'float32'), (np.expand_dims(np.arange(0,nv), axis=0) - v0[b] - nv/2))
        eta_mat = vPixelSize * np.matmul((np.expand_dims(np.arange(0,nu), axis=1) - u0[b] - nu/2), np.ones((1,nv),'float32'))
    
        if not norm:
            lt = y[:,:,b]
        elif len(norm) == 1:
            lt = -tr.log(tr.maximum(y[:,:,b]/norm[0],epsilon))
        else:
            lt = -tr.log(tr.maximum(y[:,:,b]/norm[b],epsilon))
        
        # apply filtering in v if needed
        if vFilterOn:
            pad = np.zeros((lt.shape[0],2*NV-nv))
            tmp = np.cat((lt, pad), axis=1)
            tmp = np.fft.fft(tmp, axis=1)
            tmp = np.transpose(np.tile(vFilter, (1,lt.shape[0]))) * tmp
            tmp = np.real(np.fft.ifft(tmp, axis=1))
            lt = tmp[:,0:nv]
    
        lt = lt * tr.tensor(DS[b] / np.sqrt(DS[b]**2+p_mat**2+eta_mat**2) * (A[b]/DS[b])).to(device)

        if parkerOn:
            lt = lt * tr.tile(wPker[:,b],(nv,1)).transpose(1,0).to(device)
            #lt = lt * np.tile(wPker[:,b],(nv,1))
        
        if extrapOn:
            pad = tr.zeros((2*NU-nu, lt.shape[1])).to(device)
            for j in np.arange(0,lt.shape[1]):
                pad[:,j] = tr.linspace(lt[-1,j],lt[0,j],2*NU-nu)
        else:
            pad = tr.zeros((2*NU-nu,lt.shape[1])).to(device)
    
        tmp = tr.cat((lt, pad), axis=0)
        tmp = tr.fft.fft(tmp, axis=0)
        tmp = tr.tile(filter,(1,lt.shape[1],)) * tmp
        tmp = tr.real(tr.fft.ifft(tmp,axis=0))
        y_filt[:,:,b] = tmp[0:nu,:] # un-pad
        
    dAng = np.abs(np.median(g.angle[2:]-g.angle[1:-1]))
    y_filt = y_filt * dAng/2.0 * (np.pi/180.0)

    return y_filt

# %% Multi-batch parallel filtering
## torch can automatically broadcast properly, i.e. (3,5,5) tensor can multiply by (5,5) tensor
## Only need to change the last few steps
def fbpFilterRaisedCosine2D_MB(y, fact, hamming, g, showPrg = False, **kwargs):

    (batch, nu, nv, nb) = y.shape
    device = y.device

    if 'parker' in kwargs:
        parkerOn = kwargs['parker']
    else:
        parkerOn = 0

    if 'extrap' in kwargs:
        extrapOn = kwargs['extrap']
    else:
        extrapOn = 0

    if 'vFilter' in kwargs:
        vFilterOn = kwargs['vFilter']
    else:
        vFilterOn = 0

    if 'fact_v' in kwargs:
        fact_v = kwargs['fact_v']
    else:
        fact_v = 0.8

    if 'hamming_v' in kwargs:
        hamming_v = kwargs['hamming_v']
    else:
        hamming_v = 0.5

    if 'norm' in kwargs:
        norm = kwargs['norm']
    else:
        pyf = pyyf(y)
        if pyf == 'lineInt':#'p':
            norm =[]
            print('Note: Assuming input is Line Integral (log corrected)')
        elif pyf == 'prj':
            norm =[1]
            print('Note: Assuming input is projection (no Log correction)')
        #norm = []
    
    if g.SDD.size == 1:#len(g.SDD) == 1:
        DS = np.tile(g.SDD, nb)
        A = np.tile(g.SAD, nb)
        u0 = np.tile(g.u0, nb)
        v0 = np.tile(g.v0, nb)
    else:
        DS = g.SDD
        A = g.SAD
        u0 = g.u0
        v0 = g.v0
    
    # Pixel sizes in mm
    uPixelSize = g.PixSize[0]
    vPixelSize = g.PixSize[1]

    # Filter specification in u
    NU = int(2**np.ceil(np.log2(nu)))
    filter = np.zeros((2*NU,1))
    arg1 = np.pi * fact * np.arange(0,nu) # s coordinates where the filter is calculated
    h1 = hamming * vfunc(arg1) + 0.5 * (1-hamming) * (vfunc(np.pi + arg1) + vfunc(np.pi - arg1))
    arg2 = np.pi * fact * np.arange(-nu+1, 0)
    h2 = hamming * vfunc(arg2) + 0.5 * (1-hamming) * (vfunc(np.pi + arg2) + vfunc(np.pi - arg2))
    filter[0:nu] = np.expand_dims(h1, axis=1)
    filter[(2*NU-nu+1):(2*NU)] = np.expand_dims(h2, axis=1)
    filter = 2 * (fact**2) / (2*uPixelSize*2) * filter
    filter = np.fft.fft(filter, axis=0)
    filter = tr.tensor(filter).to(device)

    # Filter specification in v (if needed)
    if vFilterOn:
        NV = int(2**np.ceil(np.log2(nv)))
        vFilter = np.zeros([2*NV,1])
        arg1 = np.arange(0, NV)
        h1 = hamming_v + (1-hamming_v) * np.cos(np.pi / (fact_v * NV) * arg1)
        if (np.ceil(fact_v*NV)+1)<len(h1):
            h1[np.ceil(fact_v*NV)+1:] = 0
        arg2 = np.arange(-NV,0)
        h2 = hamming_v + (1-hamming_v) * np.cos(np.pi / (fact_v * NV) * np.abs(arg2))
        if len(h2) > np.ceil(fact_v*NV):
            h2[0:(len(h2) - np.ceil(fact_v*NV) - 1)] = 0
        vFilter[0:NV] = np.expand_dims(h1, axis=1)
        vFilter[NV:] = np.expand_dims(h2, axis=1)

    if parkerOn:
        wPker = applyParker(nu, nv, nb, u0, DS, uPixelSize, g.angle)
        wPker = tr.tensor(wPker).to(device)

    y_filt = tr.zeros((batch,nu,nv,nb),dtype=tr.float).to(device)
    epsilon = tr.tensor(1e-10).to(device)

    if showPrg:
        iterItem = tqdm(np.arange(0, nb),desc = 'Filtering')
    else:
        iterItem = np.arange(0, nb)

    for b in iterItem:

        p_mat = uPixelSize * np.matmul(np.ones((nu,1),'float32'), (np.expand_dims(np.arange(0,nv), axis=0) - v0[b] - nv/2))
        eta_mat = vPixelSize * np.matmul((np.expand_dims(np.arange(0,nu), axis=1) - u0[b] - nu/2), np.ones((1,nv),'float32'))
    
        if not norm:
            lt = y[:,:,:,b]
        elif len(norm) == 1:
            lt = -tr.log(tr.maximum(y[:,:,:,b]/norm[0],epsilon))
        else:
            lt = -tr.log(tr.maximum(y[:,:,:,b]/norm[b],epsilon))
        
        # apply filtering in v if needed
        # disabled because it's normally not used
        # and to save time on finishing the MB code
        # if vFilterOn:
        #     pad = np.zeros((lt.shape[0],2*NV-nv))
        #     tmp = np.cat((lt, pad), axis=1)
        #     tmp = np.fft.fft(tmp, axis=1)
        #     tmp = np.transpose(np.tile(vFilter, (1,lt.shape[0]))) * tmp
        #     tmp = np.real(np.fft.ifft(tmp, axis=1))
        #     lt = tmp[:,0:nv]
    
        lt = lt * tr.tensor(DS[b] / np.sqrt(DS[b]**2+p_mat**2+eta_mat**2) * (A[b]/DS[b])).to(device)

        if parkerOn:
            lt = lt * tr.tile(wPker[:,b],(nv,1)).transpose(1,0).to(device)
            #lt = lt * np.tile(wPker[:,b],(nv,1))
        
        if extrapOn:
            pad = tr.zeros((batch, 2*NU-nu, lt.shape[2])).to(device)
            for j in np.arange(0,lt.shape[2]):
                for k in np.arange(0,lt.shape[0] ):
                    # 2 loops are probably not the most efficient way
                    pad[k,:,j] = tr.linspace(lt[k,-1,j].item(),lt[k,0,j].item(),2*NU-nu)
        else:
            pad = tr.zeros((batch, 2*NU-nu,lt.shape[2])).to(device)
        
        tmp = tr.cat((lt, pad), axis=1)
        tmp = tr.fft.fft(tmp, axis=1)
        tmp = tr.tile(filter,(1,lt.shape[2],)) * tmp
        tmp = tr.real(tr.fft.ifft(tmp,axis=1))
        y_filt[:,:,:,b] = tmp[:,0:nu,:] # un-pad
        
    dAng = np.abs(np.median(g.angle[2:]-g.angle[1:-1]))
    y_filt = y_filt * dAng/2.0 * (np.pi/180.0)

    return y_filt

#%% parker weights
def applyParker(nu, nv, nb, u0, DS, uPixelSize, ang_vec):
    
    parkIndx = np.arange(-nu/2, nu/2, dtype='float32')
    parkIndx = np.expand_dims(parkIndx, axis=1)
    u0 = np.expand_dims(u0, axis=0)
    DS = np.expand_dims(DS, axis=0)
    gamma = -np.arctan((np.tile(parkIndx,(1,nb)) - np.tile(u0,(nu,1))) * uPixelSize / np.tile(DS,(nu,1)))
    dAng = np.median(ang_vec[1:] - ang_vec[0:(len(ang_vec)-1)])
    pkerAngle = np.zeros((1,len(ang_vec)),'float32')

    # create half the parker weights
    if (dAng > 0):
        pkerAngle[0,:] = ang_vec - ang_vec[0]
    else:
        pkerAngle[0,:] = ang_vec[0] - ang_vec
        gamma = -gamma
    
    overscanAngle = ((np.abs(pkerAngle[0,-1]-pkerAngle[0,0]) * np.pi/180 - np.pi)/2)
    pkerAngle = np.tile(pkerAngle, (nu,1)) * np.pi/180

    weightPker = np.zeros(gamma.shape,'float32')
    w_map = (pkerAngle < (2 * overscanAngle - 2 * gamma))
    weightPker[w_map] = np.sin((np.pi / 4) * pkerAngle[w_map] / (overscanAngle - gamma[w_map]))**2

    w_map = ((pkerAngle >= (2 * overscanAngle - 2 * gamma)) & (pkerAngle < (np.pi - 2 * gamma)))
    weightPker[w_map] = 1

    w_map = (pkerAngle >= (np.pi - 2 * gamma)) & (pkerAngle <= (np.pi + 2 * overscanAngle))
    weightPker[w_map] = np.sin((np.pi / 4) * (np.pi + 2 * overscanAngle - pkerAngle[w_map]) / (gamma[w_map] + overscanAngle))**2

    weightPker = weightPker * 2.0

    return weightPker

# see if a projection is projection, line integral or filtered line integral
def pyyf(p):
    if p.max() > 1:
        return 'lineInt';
    elif p.min() > 0:
        return 'prj';
    else:
        return 'prjF';
# aux filter function
def vfunc(s):
    i = np.argwhere(np.abs(s) > 0.0001)
    v = np.zeros(s.shape)
    ss = s[i]
    v[i] = np.sin(ss)/ss + (np.cos(ss)-1) / (ss**2)
    v[np.abs(s) <= 0.0001] = 0.5
    return v