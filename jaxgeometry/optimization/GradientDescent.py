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

def RMGradientDescent(mu_init:Array,
                      M:object,
                      grad_fn:Callable[[Tuple[Array, Array]], Array],
                      step_size:float = 0.1,
                      max_iter:int=100,
                      bnds:Tuple[Array, Array]=(None,None),
                      max_step:Array=None
                      )->Tuple[Array, Array]:
    
    @jit
    def update(carry:Tuple[Array, Array], idx:int)->Tuple[Tuple[Array, Array],
                                                              Tuple[Array, Array]]:
        
        mu, grad = carry
        
        grad = grad #jnp.clip(grad, min_step, max_step)
        mu = M.Exp(mu, -step_size*grad)
        #mu[0] = jnp.clip(mu[0], lb, ub)
        
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
    val, carry = lax.scan(update, init=(mu_init, grad), xs=jnp.arange(0,max_iter,1))

    return val, carry #return (mu,grad),(mu_grad)

#%% Euclidean Gradient Descent

def GradientDescent(mu_init:Array,
                    grad_fn:Callable[[Tuple[Array, Array]], Array],
                    step_size:float = 0.1,
                    max_iter:int=100,
                    bnds:Tuple[Array, Array]=(None,None),
                    max_step:Array=0.1
                    )->Tuple[Array, Array]:
    
    @jit
    def update(carry:Tuple[Array, Array], idx:int)->Tuple[Tuple[Array, Array],
                                                              Tuple[Array, Array]]:
        
        mu, grad = carry
        
        grad = jnp.clip(grad, min_step/step_size, max_step/step_size)
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
    val, carry = lax.scan(update, init=(mu_init, grad), xs=jnp.arange(0,max_iter,1))

    return val, carry #return (mu,grad),(mu_grad)

#%% Joint Gradient Descent

def JointGradientDescent(mu_rm:Array,
                         mu_euc:Array,
                         M:object,
                         grad_fn_rm:Callable[[Tuple[Array, Array]], Array],
                         grad_fn_euc:Callable[[Tuple[Array, Array]], Array],
                         step_size_rm:float = 0.1,
                         step_size_euc:float = 0.1,
                         max_iter:int=100,
                         bnds_rm:Tuple[Array, Array]=(None,None),
                         bnds_euc:Tuple[Array, Array]=(None,None),
                         max_step:Array=0.1,
                         Exp = None,
                         )->Tuple[Array, Array, Array, Array]:
    
    @jit
    def update(carry:Tuple[Array, Array, Array, Array], idx:int
               )->Tuple[Tuple[Array, Array, Array, Array],
                        Tuple[Array, Array, Array, Array]]:
        
        mu_rm, mu_euc, grad_rm, grad_euc = carry
        
        grad_rm = grad_rm #jnp.clip(grad_rm, min_step, max_step)
        grad_euc = jnp.clip(grad_euc, min_step/step_size_euc, max_step/step_size_euc)
        
        mu_rm = Exp(mu_rm, -step_size_rm*grad_rm)
        #mu_rm[0] = jnp.clip(mu_rm[0], lb_rm, ub_rm)
        
        #grad_euc = grad_euc/jnp.linalg.norm(grad_euc)
        mu_euc -= step_size_euc*grad_euc
        if bool_val:
            mu_euc = jnp.clip(mu_euc, lb_euc, ub_euc)
        new_chart = M.centered_chart(mu_rm)
        mu_rm = M.update_coords(mu_rm,new_chart)
        
        grad_rm = grad_fn_rm(mu_rm, mu_euc)
        grad_euc = grad_fn_euc(mu_rm, mu_euc)
        out = (mu_rm, mu_euc, grad_rm, grad_euc)

        return out, out
        
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    
    if Exp is None:
        Exp = M.Exp
    
    lb_euc = bnds_euc[0]
    ub_euc = bnds_euc[1]
    lb_rm = bnds_rm[0]
    ub_rm = bnds_rm[1]
    
    bool_val = all(x is not None for x in bnds_euc)
    
    grad_rm = grad_fn_rm(mu_rm, mu_euc)
    grad_euc = grad_fn_euc(mu_rm, mu_euc)

    val, carry = lax.scan(update, init=(mu_rm, mu_euc, grad_rm, grad_euc), xs=jnp.arange(0,max_iter,1))
    
    return val, carry #return (mu_rm,mu_euc, grad_rm, grad_euc),(mu_rm,mu_euc, grad_rm, grad_euc)
