#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:29:55 2023

@author: fmry
"""

#%% Sources

#%%V Modules

from src.setup import *
from src.params import *

from typing import Callable

#%% GradientDescent

def GradientDescent(M:object,
                    grad_fn:Callable[[jnp.ndarray], jnp.ndarray],
                    x_init:jnp.ndarray = None,
                    step_size:float = 0.1,
                    max_iter:int=100):
    
    @jit
    def update(mu, idx):
        
        grad = grad_fn(mu)
        mu = M.Exp(mu, -step_size*grad)
        
        new_chart = M.centered_chart(mu)
        mu = M.update_coords(mu,new_chart)
    
        return mu, None
    
    if x_init is None:
        x_init = (jnp.zeros(M.dim), jnp.zeros(M.emb_dim))
        
    mu, _ = lax.scan(update, init=x_init, xs=jnp.arange(0,max_iter,1))
        
    return mu
