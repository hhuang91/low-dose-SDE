# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:51:09 2023

@author: Heyuan Huang [hhuang91@jhmi.edu]

Low dose CT projection continuous SDE
Formulated the continous transition between different doses
into SDE model fit for diffusion network training
Based partial on LDP [Wang 2014] 
Fully derivation available as PDF[https://github.com/hhuang91/low-dose-SDE/blob/main/Supplement.pdf]
"""
import torch
import numpy as np
from _utils import tensorIn_TensorOut,numpyIn_TensorOut
# tensorIn_TensorOut,numpyIn_TensorOut are decorator function to ensure input and output type

#%%
class LDSDE():
    def __init__(self, I_0:float = 1e6, I_min:float = 5e4, N:int = 101, device:str = 'cpu'):
               #(self, I_0=1e6, I_min=1e4, N=1001, device = 'cpu'):
        """Construct a Low Dose SDE.
    
        Args:
          I_0: Inital High Dose
          I_min: Targeted Lowest Dose 
          N: number of discretization steps
        """
        self.device = device
        self.sigma_e = torch.sqrt(torch.tensor(4.47).to(device))/(2**16-1)
        self.dt = torch.tensor(1/(N-1)).to(device)
        self.rev_dt = -self.dt
        self.I_0 = I_0
        self.N = N
        """Linear Dose Scheduling"""
        self.alpha = lambda t: 1 + ( I_min/I_0 - 1 )*t
        self.A = I_min/I_0 - 1
        self.D = lambda t: self.A/self.alpha(t)
        self.discrete_t = ( torch.linspace(0,N-1,N)/(N-1) ).numpy()
        self.discrete_alpha = torch.tensor(self.alpha(self.discrete_t)).numpy()
        self.discrete_D = torch.tensor(self.D(self.discrete_t)).numpy()
        self.discrete_steps = torch.linspace(0,N-1, N).long().numpy()
        
    
    @numpyIn_TensorOut
    def discretizedValues_at_T(self,T):
        step = np.interp(T, self.discrete_t, self.discrete_steps).round().astype(int)
        t = self.discrete_t[step]
        alpha = self.discrete_alpha[step]
        D = self.discrete_D[step]
        return t, alpha, D
    
    @tensorIn_TensorOut
    def getSigmaQ(self, x_neg1):
        sigma_q_sq = x_neg1/self.I_0
        return torch.sqrt(sigma_q_sq)
    
    @tensorIn_TensorOut
    def getSigma(self, sigma_q, alpha):
        sigma = torch.sqrt( alpha*sigma_q**2 + self.sigma_e**2 )
        return sigma
    
    @tensorIn_TensorOut
    def sampleXt(self, x_neg1, T):
        t, alpha, D = self.discretizedValues_at_T(T)
        sigma_q = self.getSigmaQ(x_neg1)
        sigma = self.getSigma(sigma_q, alpha)
        noise = sigma * torch.randn_like(x_neg1)
        x_t = alpha * x_neg1 + noise
        return x_t, noise
    
    @tensorIn_TensorOut
    def sampleTrainingData(self, x_neg1, T):
        t, alpha, D = self.discretizedValues_at_T(T)
        sigma_q = self.getSigmaQ(x_neg1)
        sigma = self.getSigma(sigma_q, alpha)
        noise = sigma * torch.randn_like(x_neg1)
        x_t = alpha * x_neg1 + noise
        return x_t, alpha, t
    
    @tensorIn_TensorOut
    def noiseEst2Xneg1(self, x_t ,noiseEst, alpha, x_T_Norm):
        """Note that the input noise, noiseEst, already is negative of original noise (see getNoiseGT)"""
        x_neg1_est = torch.clamp(x_T_Norm,0,1)
        sigma_q_est = self.getSigmaQ(x_neg1_est)
        sigma = self.getSigma(sigma_q_est, alpha)
        negNoise = noiseEst*sigma
        x_neg1 = (x_t + negNoise)/alpha
        return x_neg1
    
    @tensorIn_TensorOut
    def getG(self,sigma_q, D, alpha):
        g_t = ( torch.sqrt(alpha)*sigma_q + np.sqrt(2)*self.sigma_e ) * torch.sqrt(-D)
        return g_t
    
    @tensorIn_TensorOut
    def getF(self, x_t, D):
        F_xt = D * x_t
        return F_xt
    
    @tensorIn_TensorOut
    def getNoiseGT(self,x_neg1,x_t,alpha):
        """Note that noise is simply score*sigma"""
        sigma_q = self.getSigmaQ(x_neg1)
        sigma = self.getSigma(sigma_q, alpha)
        noise = -(x_t - alpha*x_neg1)/sigma
        return noise
    
    @tensorIn_TensorOut
    def getReverseSDE_dx(self, x_t, t, x_T_Norm, noiseEst):
        t, alpha, D = self.discretizedValues_at_T(t)
        x_neg1_est = torch.clamp(x_T_Norm,0,1)
        sigma_q_est = self.getSigmaQ(x_neg1_est)
        G = self.getG(sigma_q_est,D,alpha)
        F = self.getF(x_t,D)
        dw = torch.sqrt(self.dt)*torch.randn_like(x_t)
        sigma = self.getSigma(sigma_q_est, alpha)
        score = noiseEst/sigma
        dx = (F - G**2*score)*self.rev_dt + G*dw
        return dx
    
    @tensorIn_TensorOut
    def getReverseSDE_dx_GT(self, x_t, t, x_neg1, noiseEst):
        t, alpha, D = self.discretizedValues_at_T(t)
        x_neg1_est = x_neg1
        sigma_q_est = self.getSigmaQ(x_neg1_est)
        G = self.getG(sigma_q_est,D,alpha)
        F = self.getF(x_t,D)
        dw = torch.sqrt(self.dt)*torch.randn_like(x_t)
        sigma = self.getSigma(sigma_q_est, alpha)
        score = noiseEst/sigma
        dx = (F - G**2*score)*self.rev_dt + G*dw
        return dx
