#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:44:19 2023

@author: fmry
"""

#%% Sources

#%% Modules

from src.setup import *
from src.params import *

#%% Code

def initialize(M):
    
    @jit
    def hk(x,y,t):
        
        const = 1/((2*jnp.pi*t)**(M.dim*0.5))
        
        return jnp.exp(-0.5*jnp.sum(x[0]-y[0])/t)*const
    
    @jit
    def log_hk(x,y,t):
        
        return -0.5*jnp.sum(x[0]-y[0])/t-M.dim*0.5*jnp.log(2*jnp.pi*t)
    
    @jit
    def gradx_log_hk(x, y, t):
        
        return ((y[0]-x[0])/t, jnp.zeros(1))
    
    @jit
    def grady_log_hk(x, y, t):
        
        return ((x[0]-y[0])/t, jnp.zeros(1))
    
    @jit
    def gradt_log_hk(x, y, t):
        
        diff = x[0]-y[0]
        
        return 0.5*jnp.dot(diff, diff)/(t**2)-0.5*M.dim/t
    
    @jit
    def hk_mu(X_obs, t=None):

        return (jnp.mean(X_obs[0], axis=0), jnp.zeros(1))
    
    @jit
    def hk_t(X_obs, mu):

        diff_mu = X_obs[0]-mu[0]
        
        return jnp.mean(jnp.linalg.norm(diff_mu, axis = 1)**2)/M.dim
    
    @jit
    def hk_joint(X_obs):
        
        mu = opt_mu(X_obs)
        
        return mu, opt_t(X_obs, mu)
    
    M.hk = hk
    M.log_hk = log_hk
    M.gradx_log_hk = gradx_log_hk
    M.grady_log_hk = grady_log_hk
    M.gradt_log_hk = gradt_log_hk
    M.hk_mu = hk_mu
    M.hk_t = hk_t
    M.hk_joint = hk_joint
    
    return