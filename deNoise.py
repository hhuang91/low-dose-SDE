# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:30:45 2023

@author: hhuang91
"""
from _network import XDNR, loadState
from _LDSDE import LDSDE
import torch
import matplotlib.pyplot as plt
import os
from _utils import tensorIn_TensorOut
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#%% reverse function
class doseIncreaser():
    def __init__(self, stateFN = None,device = 'cpu',**kwargs):
        if stateFN is None:
            self.cnn = lambda x: (_ for _ in ()).throw(Exception('No network is loaded'))
        else:
            self.cnn = XDNR().to(device)
            loadState(self.cnn, stateFN)
            self.cnn.eval()
        self.SDE = LDSDE(device = device,**kwargs)
        self.device = device
    
    @torch.no_grad()
    @tensorIn_TensorOut
    def increaseDose(self, x_t, t, plot = False):
        x_T = x_t.clone()
        t, alphaT, _ = self.SDE.discretizedValues_at_T(t)
        x_T_Norm = x_T/alphaT
        dt = self.SDE.rev_dt
        pBar = tqdm(total = int(torch.round(t/self.SDE.dt)),desc = 'Denoising steps:')
        while (t>0):
            t , alpha, D = self.SDE.discretizedValues_at_T(t)
            noise_est = self.cnn((x_T/alphaT).view(-1,1,*x_T.shape[-2:]),
                              (x_t/alpha).view(-1,1,*x_t.shape[-2:]),
                              t.view(-1,1))
            dx = self.SDE.getReverseSDE_dx(x_t, t, x_T_Norm, noise_est)
            if plot:
                plt.figure()
                plt.subplot(2,1,1)
                plt.imshow(x_t[0,0,:,:].detach().cpu().numpy(),cmap='gray');plt.colorbar()
                plt.subplot(2,1,2)
                plt.imshow((noise_est[0,0,:,:]).detach().cpu().numpy(),cmap='gray');plt.colorbar()
                plt.show()
            x_t = x_t + dx
            t = t + dt
            pBar.update(1)
        return x_t
    
    @torch.no_grad()
    @tensorIn_TensorOut
    def removeNoise(self,x_t, t, nIter = 20, **kwargs):
        res = 0
        for i in range(nIter):
            print(f'{i+1}/{nIter}')
            x_dN = self.increaseDose(x_t, t, **kwargs)
            res+=x_dN
        res/=nIter
        return res
    


class doseIncreaserGT():
    def __init__(self,device = 'cpu'):
        self.SDE = LDSDE(device = device)
        self.device = device
    
    @torch.no_grad()
    @tensorIn_TensorOut
    def doseIncreaserGTtest(self, x_t, t, x_neg1, plot=False):
        x_T = x_t.clone()
        t, alphaT, _ = self.SDE.discretizedValues_at_T(t)
        x_T_Norm = x_T/alphaT
        dt = self.SDE.rev_dt
        pBar = tqdm(total = int(torch.round(t/self.SDE.dt)),desc = 'Denoising steps:')
        while (t>0):
            t , alpha, D = self.SDE.discretizedValues_at_T(t)
            noise_est = self.SDE.getNoiseGT(x_neg1,x_t,alpha)
            dx = self.SDE.getReverseSDE_dx(x_t, t, x_T_Norm, noise_est)
            if plot:
               plt.figure()
               plt.subplot(2,1,1)
               plt.imshow((x_t[0,0,:,:]/alpha).detach().cpu().numpy(),cmap='gray');plt.colorbar()
               plt.subplot(2,1,2)
               plt.imshow((noise_est[0,0,:,:]).detach().cpu().numpy(),cmap='gray');plt.colorbar()
               plt.show()
            x_t = x_t + dx
            t = t + dt
            pBar.update(1)
        return x_t
    
    @torch.no_grad()
    @tensorIn_TensorOut
    def doseIncreaserGTtest_trueGT(self, x_t, t, x_neg1, plot=False):
        dt = self.SDE.rev_dt
        pBar = tqdm(total = int(torch.round(t/self.SDE.dt)),desc = 'Denoising steps:')
        while (t>0):
            t , alpha, D = self.SDE.discretizedValues_at_T(t)
            noise_est = self.SDE.getNoiseGT(x_neg1,x_t,alpha)
            dx = self.SDE.getReverseSDE_dx_GT(x_t, t, x_neg1, noise_est)
            if plot:
               plt.figure()
               plt.subplot(2,1,1)
               plt.imshow((x_t[0,0,:,:]/alpha).detach().cpu().numpy(),cmap='gray');plt.colorbar()
               plt.subplot(2,1,2)
               plt.imshow((noise_est[0,0,:,:]).detach().cpu().numpy(),cmap='gray');plt.colorbar()
               plt.show()
            x_t = x_t + dx
            t = t + dt
            pBar.update(1)
        return x_t
    
    # @torch.no_grad()
    # @tensorIn_TensorOut
    # def increaseDose(self, x_t, t, plot = False):
    #     x_T = x_t.clone()
    #     t, alphaT, _ = self.SDE.discretizedValues_at_T(t)
    #     dt = self.SDE.rev_dt
    #     pBar = tqdm(total = int(torch.round(t/self.SDE.dt)),desc = 'Denoising steps:')
    #     while (t>0):
    #         t , alpha, D = self.SDE.discretizedValues_at_T(t)
    #         score_est = self.cnn(x_T.view(-1,1,*x_T.shape[-2:]),
    #                          x_t.view(-1,1,*x_t.shape[-2:]),
    #                          t.view(-1,1))
    #         dx = self.SDE.getReverseSDE_dx(x_t, t, score_est)
    #         if plot:
    #             plt.imshow(x_t.cpu().numpy().squeeze(), cmap='gray')
    #             plt.title(f't = {t}')
    #             plt.colorbar()
    #             plt.show()
    #         x_t = x_t + dx
    #         t = t + dt
    #         pBar.update(1)
    #     return x_t
    
    # @torch.no_grad()
    # @tensorIn_TensorOut
    # def increaseDose_anyXneg1(self,x_t, t, x_neg1, plot = False):
    #     x_T = x_t.clone()
    #     t, alphaT, _ = self.SDE.discretizedValues_at_T(t)
    #     dt = self.SDE.rev_dt
    #     pBar = tqdm(total = int(torch.round(t/self.SDE.dt)),desc = 'Denoising steps:')
    #     while (t>0):
    #         t , alpha, D = self.SDE.discretizedValues_at_T(t)
    #         score_est = self.cnn(x_T.view(-1,1,*x_T.shape[-2:]),
    #                          x_t.view(-1,1,*x_t.shape[-2:]),
    #                          t.view(-1,1))
    #         dx = self.SDE.getReverseSDE_dx_with_Xneg1(x_t, t, x_neg1, score_est)
    #         if plot:
    #             plt.imshow(x_t.cpu().numpy().squeeze(), cmap='gray')
    #             plt.title(f't = {t}')
    #             plt.colorbar()
    #             plt.show()
    #         x_t = x_t + dx
    #         t = t + dt
    #         pBar.update(1)
    #     return x_t
    # @torch.no_grad()
    # @tensorIn_TensorOut
    # def doseIncreaserGTtest(self, x_t, t, x_neg1, plot=False):
    #     dt = self.SDE.rev_dt
    #     pBar = tqdm(total = int(torch.round(t/self.SDE.dt)),desc = 'Denoising steps:')
    #     while (t>0):
    #         t , alpha, D = self.SDE.discretizedValues_at_T(t)
    #         score = self.SDE.getScoreGT(x_neg1,x_t,alpha)
    #         dx = self.SDE.getReverseSDE_dx(x_t, t, score)
    #         if plot:
    #             plt.imshow(x_t.cpu().numpy().squeeze(), cmap='gray')
    #             plt.title(f't = {t}')
    #             plt.colorbar()
    #             plt.show()
    #         x_t = x_t + dx
    #         t = t + dt
    #         pBar.update(1)
    #     return x_t
    
    # @torch.no_grad()
    # @tensorIn_TensorOut
    # def doseIncreaserTest_with_est_xNeg1(self, x_t, t, x_neg1, x_neg1_est, plot=False):
    #     dt = self.SDE.rev_dt
    #     pBar = tqdm(total = int(torch.round(t/self.SDE.dt)),desc = 'Denoising steps:')
    #     while (t>0):
    #         t , alpha, D = self.SDE.discretizedValues_at_T(t)
    #         score = self.SDE.getScoreGT(x_neg1,x_t,alpha)
    #         dx = self.SDE.getReverseSDE_dx_with_Xneg1(x_t, t, x_neg1_est, score)
    #         if plot:
    #             plt.imshow(x_t.cpu().numpy().squeeze(), cmap='gray')
    #             plt.title(f't = {t}')
    #             plt.colorbar()
    #             plt.show()
    #         x_t = x_t + dx
    #         t = t + dt
    #         pBar.update(1)
    #     return x_t
    
    # @torch.no_grad()
    # @tensorIn_TensorOut
    # def doseIncreaserTest_with_noise_est(self, x_t, t, x_neg1, plot=False):
    #     x_T = x_t.clone()
    #     dt = self.SDE.rev_dt
    #     pBar = tqdm(total = int(torch.round(t/self.SDE.dt)),desc = 'Denoising steps:')
    #     while (t>0):
    #         t , alpha, D = self.SDE.discretizedValues_at_T(t)
    #         noise_est = self.SDE.getNoiseGT(x_neg1,x_t,alpha)
    #         dx = self.SDE.getReverseSDE_dx_with_noise_est(x_t, t, x_T, noise_est)
    #         if plot:
    #             plt.imshow(x_t.cpu().numpy().squeeze(), cmap='gray')
    #             plt.title(f't = {t}')
    #             plt.colorbar()
    #             plt.show()
    #         x_t = x_t + dx
    #         t = t + dt
    #         pBar.update(1)
    #     return x_t
    
