#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:29:55 2023

@author: fmry
"""

#%% Sources

#%%V Modules

from jaxgeometry.setup import *

#%% Riemannian Gradient Descent

@jit
def RMGradientDescent(mu_init:ndarray,
                      M:object,
                      grad_fn:Callable[[tuple[ndarray, ndarray]], ndarray],
                      step_size:float = 0.1,
                      max_iter:int=100,
                      bnds:tuple[ndarray, ndarray]=(None,None),
                      max_step:ndarray=None
                      )->tuple[ndarray, ndarray]:
    
    @jit
    def update(carry:tuple[ndarray, ndarray], idx:int)->tuple[tuple[ndarray, ndarray],
                                                              tuple[ndarray, ndarray]]:
        
        mu, grad = carry
        
        grad = jnp.clip(grad, min_step, max_step)
        mu = M.Exp(mu, -step_size*grad)
        mu[0] = jnp.clip(mu[0], lb, ub)
        
        new_chart = M.centered_chart(mu)
        mu = M.update_coords(mu,new_chart)
        
        grad = grad_fn(mu)
        out = (mu, grad)
    
        return out, out
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb = bnds[0]
    ub = bnds[1]
        
    grad = grad_fn(mu_init)
    _, out = lax.scan(update, init=(mu_init, grad), xs=jnp.arange(0,max_iter,1))
    
    mu = out[0]
    grad = out[1]    
    
    return mu, grad

#%% Euclidean Gradient Descent

@jit
def GradientDescent(mu_init:ndarray,
                    grad_fn:Callable[[tuple[ndarray, ndarray]], ndarray],
                    step_size:float = 0.1,
                    max_iter:int=100,
                    bnds:tuple[ndarray, ndarray]=(None,None),
                    max_step:ndarray=None
                    )->tuple[ndarray, ndarray]:
    
    @jit
    def update(carry:tuple[ndarray, ndarray], idx:int)->tuple[tuple[ndarray, ndarray],
                                                              tuple[ndarray, ndarray]]:
        
        mu, grad = carry
        
        grad = jnp.clip(grad, min_step, max_step)
        mu -= step_size*grad
        mu = jnp.clip(mu, lb, ub)
        
        grad = grad_fn(mu)
        out = (mu, grad)
    
        return out, out
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb = bnds[0]
    ub = bnds[1]
        
    grad = grad_fn(mu_init)
    _, out = lax.scan(update, init=(mu_init, grad), xs=jnp.arange(0,max_iter,1))
    
    mu = out[0]
    grad = out[1]

    return mu, grad

#%% Joint Gradient Descent

@jit
def JointGradientDescent(mu_rm:ndarray,
                         mu_euc:ndarray,
                         M:object,
                         grad_fn_rm:Callable[[tuple[ndarray, ndarray]], ndarray],
                         grad_fn_euc:Callable[[tuple[ndarray, ndarray]], ndarray],
                         step_size_rm:float = 0.1,
                         step_size_euc:float = 0.1,
                         max_iter:int=100,
                         bnds_rm:tuple[ndarray, ndarray]=(None,None),
                         bnds_euc:tuple[ndarray, ndarray]=(None,None),
                         max_step:ndarray=None
                         )->tuple[ndarray, ndarray, ndarray, ndarray]:
    
    @jit
    def update(carry:tuple[ndarray, ndarray, ndarray, ndarray], idx:int
               )->tuple[tuple[ndarray, ndarray, ndarray, ndarray],
                        tuple[ndarray, ndarray, ndarray, ndarray]]:
        
        mu_rm, mu_euc, grad_rm, grad_euc = carry
        
        grad_rm = jnp.clip(grad_rm, min_step, max_step)
        grad_euc = jnp.clip(grad_euc, min_step, max_step)
        
        mu_rm = M.Exp(mu, -step_size_rm*grad_rm)
        mu_rm[0] = jnp.clip(mu_rm[0], lb_rm, ub_rm)
        
        mu_euc -= step_size_euc*grad_euc
        mu_euc = jnp.clip(mu_euc, lb_euc, ub_euc)
    
        new_chart = M.centered_chart(mu_rm)
        mu_rm = M.update_coords(mu_rm,new_chart)
        
        grad_rm = grad_fn_rm(mu_rm)
        grad_euc = grad_fn_euc(mu_euc)
        out = (mu_rm, mu_euc, grad_rm, grad_euc)
    
        return out, out
        
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb_euc = bnds_euc[0]
    ub_euc = bnds_euc[1]
    lb_rm = bnds_rm[0]
    ub_rm = bnds_rm[1]
    
    grad_rm = grad_fn_rm(mu_rm)
    grad_euc = grad_fn_euc(mu_euc)
    _, out = lax.scan(update, init=(mu_rm, mu_euc, grad_rm, grad_euc), xs=jnp.arange(0,max_iter,1))
    
    mu_rm = out[0]
    mu_euc = out[1]
    grad_rm = out[2]
    grad_euc = out[3]
    
    return mu_rm, mu_euc, grad_rm, grad_euc
