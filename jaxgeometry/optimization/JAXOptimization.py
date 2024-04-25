#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:04:19 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Riemannian Jax Optimization

def RMJaxOpt(mu_init:Array,
             M:object,
             grad_fn:Callable[[Tuple[Array, Array]], Array],
             max_iter:int=100,
             optimizer:Callable=None,
             lr_rate:float = 0.0002,
             bnds:Tuple[Array, Array]=(None, None),
             max_step:Array=None
             )->Tuple[Array, Array]:
    
    @jit
    def update(carry:Tuple[Array, Array, object], idx:int
               )->Tuple[Tuple[Array, Array, object],
                        Tuple[Array, Array]]:
        
        mu, grad, opt_state = carry
        
        opt_state = opt_update(idx, grad, opt_state)
        mu_rm = get_params(opt_state)
        
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
        opt_init, opt_update, get_params = optimizers.sgd(lr_rate)
    else:
        opt_init, opt_update, get_params = optimizer
        
    opt_state = opt_init(mu_init[0])
    grad = grad_fn(mu_init)
    val, carry = lax.scan(update, init = (mu_init, grad, opt_state), xs = jnp.arange(0,max_iter,1))
    
    return val, carry #(mu,grad), (mu,grad)

#%% Euclidean Jax Optimization

def JaxOpt(mu_init:Array,
           M:object,
           grad_fn:Callable[[Array], Array],
           max_iter:int=100,
           optimizer:Callable=None,
           lr_rate:float = 0.0002,
           bnds:Tuple[Array, Array]=(None,None),
           max_step=None
           )->Tuple[Array, Array]:
    
    @jit
    def update(carry:Tuple[Array, Array, object], idx:int
               )->Tuple[Tuple[Array, Array, object],
                        Tuple[Array, Array]]:
        
        mu, grad, opt_state = carry
        
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
        opt_init, opt_update, get_params = optimizers.sgd(lr_rate)
    else:
        opt_init, opt_update, get_params = optimizer
        
    opt_state = opt_init(mu_init)
    grad = grad_fn(mu_init)
    val, carry = lax.scan(update, init = (mu_init, grad, opt_state), xs = jnp.arange(0,max_iter,1))
    
    return val, carry #(mu,grad), (mu,grad)

#%% Joint Jax Optimization

def JointJaxOpt(mu_rm:Array,
                mu_euc:Array,
                M:object,
                grad_fn_rm:Callable[[Tuple[Array, Array]], Array],
                grad_fn_euc:Callable[[Array], Array],
                max_iter:int=100,
                optimizer:Callable=None,
                lr_rate:float=0.0002,
                bnds_rm:Tuple[Array, Array]=(None,None),
                bnds_euc:Tuple[Array, Array]=(None,None),
                max_step:float=0.1
                )->Tuple[Array, Array, Array, Array]:
    
    @jit
    def update(carry:Tuple[Array, Array, Array, Array, object], 
               idx:int
               )->Tuple[Tuple[Array, Array, Array, Array, object],
                        Tuple[Array, Array, Array, Array]]:
        
        mu_rm, mu_euc, grad_rm, grad_euc, opt_state = carry
        
        grad_rm = grad_rm #jnp.clip(grad_rm, min_step, max_step)
        grad_euc = jnp.clip(grad_euc, min_step, max_step)
        
        grad = jnp.hstack((grad_rm, grad_euc))
        opt_state = opt_update(idx, grad, opt_state)
        mu = get_params(opt_state)
        
        mux_rm = mu[:N_rm]
        mu_euc = mu[N_rm:]
        
        mux_rm = mux_rm#jnp.clip(mux_rm, lb_rm, ub_rm)
        if bool_val:
            mu_euc = jnp.clip(mu_euc, lb_euc, ub_euc)
        
        new_chart = M.centered_chart((mux_rm, mu_rm[1]))
        mu_rm = M.update_coords((mux_rm, mu_rm[1]),new_chart)
        
        grad_rm = grad_fn_rm(mu_rm, mu_euc)
        grad_euc = grad_fn_euc(mu_rm, mu_euc)
        
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
        opt_init, opt_update, get_params = optimizers.sgd(lr_rate)
    else:
        opt_init, opt_update, get_params = optimizer
        
    bool_val = all(x is not None for x in bnds_euc)
        
    opt_state = opt_init(jnp.hstack((mu_rm[0], mu_euc)))
    grad_rm = grad_fn_rm(mu_rm, mu_euc)
    grad_euc = grad_fn_euc(mu_rm, mu_euc)
    N_rm = len(grad_rm)
    val, carry = lax.scan(update, init = (mu_rm, mu_euc, 
                                          grad_rm, grad_euc, 
                                          opt_state), xs = jnp.arange(0,max_iter,1))
    
    return val, carry #(mu_rm, mu_euc, grad_rm, grad_euc), (mu_rm, mu_euc, grad_rm, grad_euc)
