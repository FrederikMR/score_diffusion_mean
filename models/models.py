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
from jax import vmap

#haiku
import haiku as hk

#dataclasses
import dataclasses

#%% MLP Score
@dataclasses.dataclass
class MLP_s1(hk.Module):
    
    dim:int
    layers:list
    
    def model(self)->object:
        
        model = []
        for l in self.layers:
            model.append(hk.Linear(l))
            model.append(tanh)
            
        model.append(hk.Linear(self.dim))
        
        return hk.Sequential(model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        x_new = x.T
        x1 = x_new[:self.dim].T
        x2 = x_new[self.dim:(2*self.dim)].T
        #t = x_new[-1]
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x_new[-1].reshape(shape)
            
        grad_euc = (x1-x2)/t
      
        return self.model()(x)+grad_euc
    
@dataclasses.dataclass
class MLP_s2(hk.Module):
    
    layers_alpha:list
    layers_beta:list
    dim:int = 2
    r:int = max(dim // 2,1)
    
    def model_alpha(self)->object:
        
        model = []
        for l in self.layers_alpha:
            model.append(hk.Linear(l))
            model.append(tanh)
            
        model.append(hk.Linear(self.dim))
        
        return hk.Sequential(model)
    
    def model_beta(self)->object:
        
        model = []
        for l in self.layers_beta:
            model.append(hk.Linear(l))
            model.append(tanh)
            
        model.append(lambda x: hk.Linear(self.dim*self.r)(x).reshape(-1,self.dim,self.r))
        
        return hk.Sequential(model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        alpha = self.model_alpha()(x).reshape(-1,self.dim)
        diag = vmap(lambda x: jnp.diag(x))(alpha)
        #beta = self.model_beta()(x)
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x.T[-1].reshape(shape)

        hess_rn = -jnp.einsum('ij,...i->...ij', jnp.eye(self.dim), 1/t)
        #diag = vmap(lambda x,t: jnp.diag(x/t))(alpha, t)
        
        return (diag+hess_rn).squeeze()#+jnp.einsum('...ik,...jk->...ij', beta, beta).squeeze()#+\
            
@dataclasses.dataclass
class MLP_s1s2(hk.Module):
    
    s1_model:MLP_s1
    s2_model:MLP_s2

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        return self.s1_model(x), self.s2_model(x)