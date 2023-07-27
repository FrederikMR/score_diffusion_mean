#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:55:46 2023

@author: fmry
"""

#%% Sources

#%% Modules

from src.setup import *
from src.params import *

from typing import Callable

#%% Gradient-Newton Method

def GradientNewton(M:object,
                  grad_fn:Callable[[jnp.ndarray], jnp.ndarray],
                  ggrad_fn:Callable[[jnp.ndarray], jnp.ndarray] = None,
                  x_init:jnp.ndarray = None,
                  grad_step:float = 0.1,
                  newton_step:float=0.1,
                  iter_step:int = 10,
                  tol = 1e-1,
                  max_iter:int=100):
    
    @jit
    def update_gradient(mu, idx):
        
        grad = grad_fn(mu)
        mu = M.Exp(mu, -step_size*grad)
        
        new_chart = M.centered_chart(mu)
        mu = M.update_coords(mu,new_chart)
    
        return mu, None
    
    @jit
    def update_newton(mu, idx):
        
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
        
    step_grid = jnp.arange(0, iter_step, 1)
    max_grid = jnp.arange(0, max_iter, 1)        
    
    for i in range(max_iter):
        mu, _ = lax.scan(update_gradient, init=mu, xs=step_grid)
        if jnp.linalg.norm(grad_fn(mu)) < tol:
            mu, _ = lax.scan(update_newton, init=mu, xs=step_grid)
    
    return mu