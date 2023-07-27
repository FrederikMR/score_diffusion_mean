#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:50:38 2023

@author: fmry
"""

#%% Sources

#%% Modules

from src.setup import *
from src.params import *

from typing import Callable

#%% Newton's Method

def NewtonsMethod(M:object,
                  grad_fn:Callable[[jnp.ndarray], jnp.ndarray],
                  ggrad_fn:Callable[[jnp.ndarray], jnp.ndarray] = None,
                  x_init:jnp.ndarray = None,
                  step_size:float = 0.1,
                  max_iter:int=100):
    
    @jit
    def update(mu, idx):
        
        grad = grad_fn(mu)
        ggrad = ggrad_fn(mu)
        
        step_grad = jscipy.sparse.linalg.gmres(ggrad, grad)[0]
        
        mu = M.Exp(mu, -step_size*step_grad)
        
        new_chart = M.centered_chart(mu)
        mu = M.update_coords(mu,new_chart)
    
        return mu, None
    
    if x_init is None:
        x_init = (jnp.zeros(M.dim), jnp.zeros(M.emb_dim))
        
    if ggrad_fn is None:
        ggrad_fn = jacfwdx(grad_fn)
        
    mu, _ = lax.scan(update, init=x_init, xs=jnp.arange(0,max_iter,1))
        
    return mu









