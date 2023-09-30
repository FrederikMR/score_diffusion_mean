#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:11:07 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax
import jax.numpy as jnp
from jax.nn import tanh

#haiku
import haiku as hk

#dataclasses
import dataclasses

#%% Models

@dataclasses.dataclass
class s1_model(hk.Module):
    
    dim:int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        x_new = x.T
        x1 = x_new[:self.dim].T
        x2 = x_new[self.dim:(2*self.dim)].T
        t = x_new[-1]
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x_new[-1].reshape(shape)
            
        grad_euc = (x1-x2)/t
        model = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            hk.Linear(200), tanh,
            hk.Linear(400), tanh,
            hk.Linear(400), tanh,
            hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            lambda x: hk.Linear(self.dim)(x)
            ])
      
        return model(x)+grad_euc
    
@dataclasses.dataclass
class s2_model(hk.Module):
    
    dim:int = 2
    r:int = max(dim // 2,1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        model_alpha = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            hk.Linear(200), tanh,
            hk.Linear(400), tanh,
            hk.Linear(400), tanh,
            hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            hk.Linear(self.dim)
            ])
        
        model_beta = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            hk.Linear(200), tanh,
            hk.Linear(400), tanh,
            hk.Linear(400), tanh,
            hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            lambda x: hk.Linear(self.dim*self.r)(x).reshape(-1,self.dim, self.r)
            ])
        
        beta = model_beta(x)
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x.T[-1].reshape(shape)

        hess_rn = -jnp.einsum('ij,...i->...ij', jnp.eye(self.dim), 1/t)
        
        return jnp.diag(model_alpha(x))+jnp.einsum('...ik,...jk->...ij', beta, beta)+\
            hess_rn