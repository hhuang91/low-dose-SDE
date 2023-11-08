# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:06:09 2023
Distributed Data Parallel warpper for training the network
@author: hhuang91
"""

from __future__ import print_function
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.optim as optim
import scipy
from _dataLoader import hdf5LoaderDDP
from _structs import gStruct
from _network import XDNR
from _trainNetwork import trainNTest

def Train(  args:dict,
            deviceN:str,
            gLoc:str,
            lr:float,
            wMSE:float, wFbpMSE:float,
            continueTrain:bool = False, xferLearn:bool = False,
            prjSize:List[int] = [1024,360],recDim:int = 512,recVoxSize:float = 0.25,
            EPOCHS:int = 500,batchSize:int = 8,
            dataPath:str = './Data',outDir:str = './networkOutput/',
            numWorker:int = 0,
            dispOn:bool = False):
    # setup DDP env
    torch.cuda.set_device(args.local_rank)
    world_size = args.ngpu
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=args.local_rank,
        )
    # set up device
    device = torch.device('cuda:{}'.format(args.local_rank))
    # get geometry
    tmp = scipy.io.loadmat(gLoc)
    gRaw = tmp['g']
    g = gStruct()
    g.setFromSioLoaded(gRaw)
    # set up network for DDP 
    cnn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(XDNR())
    cnn = cnn.to(device)
    cnn = torch.nn.parallel.DistributedDataParallel(cnn,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)
    # optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    # data loader                                    
    dataLoader = hdf5LoaderDDP(dataPath,batchSize,True,args,numWorker)
    # setup training object
    tNt = trainNTest(   device,
                        g,
                        cnn,
                        optimizer,wMSE,wFbpMSE,
                        continueTrain,xferLearn,
                        prjSize,recDim,recVoxSize,
                        EPOCHS,
                        dataLoader,outDir,
                        dispOn,
                        rank = args.local_rank)
    # TRAIN
    tNt.train()