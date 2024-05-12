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
from jax.nn import tanh, sigmoid

#haiku
import haiku as hk

#dataclasses
import dataclasses

#%% Models

@dataclasses.dataclass
class MLP_f(hk.Module):
    
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
        
        return self.model()(x)
    
@dataclasses.dataclass
class MLP_sigma(hk.Module):

    layers:list
    
    def model(self)->object:
        
        model = []
        for l in self.layers:
            model.append(hk.Linear(l))
            model.append(tanh)
            
        model.append(hk.Linear(1))
        
        return hk.Sequential(model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        return sigmoid(self.model()(x)).squeeze()
            
@dataclasses.dataclass
class MLP_mlnr(hk.Module):
    
    f:MLP_f
    sigma:MLP_sigma

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        return self.f(x), self.sigma(x)