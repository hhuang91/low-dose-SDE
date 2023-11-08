# -*- coding: utf-8 -*-
"""
Use LDSDE to denoise breast CT projection data
by diffuse low-dose projections to high-dose projections

@author: hhuang91
"""
from _network import XDNR, loadState
from _LDSDE import LDSDE
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tqdm import tqdm
from _utils import tensorIn_TensorOut
#tensorIn_TensorOut decorator will ensure the input and output be tensors

#%% reverse function
class doseIncreaser():
    """
        Use trained network for diffusion
    """
    def __init__(self, stateFN:str = None,device:str = 'cpu',**kwargs):
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
    


class doseIncreaserGT():
    """
        Test if SDE works using ground truth score for each step
    """
    def __init__(self,device:str = 'cpu'):
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
    
    
