#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:04:19 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JAXGeometry.setup import *
from JAXGeometry.params import *

#%% Riemannian Jax Optimization

@jit
def RMJaxOpt(mu_init:jnp.ndarray,
             M:object,
             grad_fn:Callable[[jnp.ndarray], jnp.ndarray],
             step_size:float = 0.1,
             max_iter:int=100,
             optimizer:Callable=None,
             opt_params:tuple=(0.1, 0.9, 0.999, 1e-8),
             bnds:tuple[jnp.ndarray, jnp.ndarray]=(None, None),
             max_step:jnp.ndarray=None
             )->tuple[jnp.ndarray, jnp.ndarray]:
    
    @jit
    def update(carry:tuple[jnp.ndarray, jnp.ndarray, object], idx:int
               )->tuple[tuple[jnp.ndarray, jnp.ndarray, object],
                        tuple[jnp.ndarray, jnp.ndarray]]:
        
        mu, grad, opt_state = carry
        
        grad = jnp.clip(grad, min_step, max_step)
        
        opt_state = opt_update(idx, grad, opt_state)
        mu_rm = get_params(opt_state)
        mu_rm = jnp.clip(mu_rm, lb, ub)
        
        new_chart = M.centered_chart((mu_rm, mu[1]))
        mu = M.update_coords((mu_rm, mu[1]),new_chart)
        
        grad = grad_fn(mu)
        
        return (mu, grad, opt_state), (mu, grad)
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb = bnds[0]
    ub = bnds[1]
    
    if optimizer is None:
        optimizer = optimizers.adam
        opt_init, opt_update, get_params = optimizer(0.1, b1=0.9, b2=0.999, eps=1e-8)
    else:
        opt_init, opt_update, get_params = optimizer(*opt_params)
        
    opt_state = opt_init(mu_init[0])
    grad = grad_fn(mu_init)
    _, out = lax.scan(update, init = (mu_init, grad, opt_state), xs = jnp.arange(0,max_iter,1))
    mu = out[0]
    grad = out[1]
    
    return mu, grad

#%% Euclidean Jax Optimization

@jit
def JaxOpt(mu_init:jnp.ndarray,
           M:object,
           grad_fn:Callable[[jnp.ndarray], jnp.ndarray],
           step_size:float = 0.1,
           max_iter:int=100,
           optimizer:Callable=None,
           opt_params:tuple=(0.1, 0.9, 0.999, 1e-8),
           bnds:tuple[jnp.ndarray, jnp.ndarray]=(None,None),
           max_step=None
           )->tuple[jnp.ndarray, jnp.ndarray]:
    
    @jit
    def update(carry:tuple[jnp.ndarray, jnp.ndarray, object], idx:int
               )->tuple[tuple[jnp.ndarray, jnp.ndarray, object],
                        tuple[jnp.ndarray, jnp.ndarray]]:
        
        mu, grad, opt_state = carry
        
        grad = jnp.clip(grad, min_step, max_step)
        
        opt_state = opt_update(idx, grad, opt_state)
        mu = get_params(opt_state)
        
        mu = jnp.clip(mu, lb, ub)
        
        grad = grad_fn(mu)
        
        return (mu, grad, opt_state), (mu, grad)
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb = bnds[0]
    ub = bnds[1]
    
    if optimizer is None:
        optimizer = optimizers.adam
        opt_init, opt_update, get_params = optimizer(0.1, b1=0.9, b2=0.999, eps=1e-8)
    else:
        opt_init, opt_update, get_params = optimizer(*opt_params)
        
    opt_state = opt_init(mu_init)
    grad = grad_fn(mu_init)
    _, out = lax.scan(update, init = (mu_init, grad, opt_state), xs = jnp.arange(0,max_iter,1))
    mu = out[0]
    grad = out[1]
    
    return mu, grad

#%% Joint Jax Optimization

@jit
def JointJaxOpt(mu_rm:jnp.ndarray,
                mu_euc:jnp.ndarray,
                M:object,
                grad_fn_rm:Callable[[jnp.ndarray], jnp.ndarray],
                grad_fn_euc:Callable[[jnp.ndarray], jnp.ndarray],
                max_iter:int=100,
                optimizer:Callable=None,
                opt_params:tuple=(0.1, 0.9, 0.999, 1e-8),
                bnds_rm:tuple[jnp.ndarray, jnp.ndarray]=(None,None),
                bnds_euc:tuple[jnp.ndarray, jnp.ndarray]=(None,None),
                max_step:jnp.ndarray=None
                )->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    @jit
    def update(carry:tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, object], 
               idx:int
               )->tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, object],
                        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        
        mu_rm, mu_euc, grad_rm, grad_euc, opt_state = carry
        
        grad_rm = jnp.clip(grad_rm, min_step, max_step)
        grad_euc = jnp.clip(grad_euc, min_step, max_step)
        
        grad = jnp.hstack((grad_rm, grad_euc))
        opt_state = opt_update(idx, grad, opt_state)
        mu = get_params(opt_state)
        
        mux_rm = mu[:N_rm]
        mu_euc = mu[N_rm:]
        
        mux_rm = jnp.clip(mux_rm, lb_rm, ub_rm)
        mu_euc = jnp.clip(mu_euc, lb_euc, ub_euc)
        
        new_chart = M.centered_chart((mux_rm, mu_rm[1]))
        mu_rm = M.update_coords((mux_rm, mu_rm[1]),new_chart)
        
        grad_rm = grad_fn_rm(mu_rm)
        grad_euc = grad_fn_euc(mu_euc)
        
        return (mu_rm, mu_euc, grad_rm, grad_euc, opt_state), \
            (mu_rm, mu_euc, grad_rm, grad_euc)
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb_euc = bnds_euc[0]
    ub_euc = bnds_euc[1]
    lb_rm = bnds_rm[0]
    ub_rm = bnds_rm[1]
    
    if optimizer is None:
        optimizer = optimizers.adam
        opt_init, opt_update, get_params = optimizer(0.1, b1=0.9, b2=0.999, eps=1e-8)
    else:
        opt_init, opt_update, get_params = optimizer(*opt_params)
        
    opt_state = opt_init(jnp.hstack((mu_rm[0], mu_euc)))
    grad_rm = grad_fn_rm(mu_rm)
    grad_euc = grad_fn_euc(mu_euc)
    N_rm = len(grad_rm)
    _, out = lax.scan(update, init = (mu_rm, mu_euc, 
                                      grad_rm, grad_euc, 
                                      opt_state), xs = jnp.arange(0,max_iter,1))
    mu = out[0]
    grad = out[1]
    
    return mu, grad

