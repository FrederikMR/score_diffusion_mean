#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:50:38 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Riemannian Newton's Method

@jit
def RMNewtonsMethod(mu_init:Array,
                    M:object,   
                    grad_fn:Callable[[Tuple[Array, Array]], Array],
                    ggrad_fn:Callable[[Tuple[Array, Array]], Array] = None,
                    step_size:float = 0.1,
                    max_iter:int=100,
                    bnds:tuple[Array, Array]=(None, None),
                    max_step:Array=None
                    )->Tuple[Tuple[Array, Array], Array]:
    
    @jit
    def update(carry:Tuple[Array, Array], idx:int
               )->Tuple[Tuple[Array, Array],
                        Tuple[Array, Array]]:
        
        mu, grad = carry
        
        ggrad = ggrad_fn(mu)
        
        step_grad = jscipy.sparse.linalg.gmres(ggrad, grad)[0]
        step_grad = jnp.clip(step_grad, min_step, max_step)
        
        mu = M.Exp(mu, -step_size*step_grad)
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
        
    if ggrad_fn is None:
        ggrad_fn = jacfwdx(grad_fn)
        
    grad = grad_fn_rm(mu_init)
    val, carry = lax.scan(update, init=(mu_init, grad), xs=jnp.arange(0,max_iter,1))

    return val, carry #(mu, grad), (mu, grad)

#%% Euclidean Newton's Method

@jit
def NewtonsMethod(mu_init:Array,
                  grad_fn:Callable[[Array], Array],
                  ggrad_fn:Callable[[Tuple[Array, Array]], Array] = None,
                  step_size:float = 0.1,
                  max_iter:int=100,
                  bnds:Tuple[Array, Array]=(None,None),
                  max_step:Array=None
                  )->Tuple[Array, Array]:
    
    @jit
    def update(carry:Tuple[Array, Array], idx:int
               )->Tuple[Tuple[Array, Array],
                        Tuple[Array, Array]]:
        
        mu, grad = carry
        
        ggrad = ggrad_fn(mu)
        
        step_grad = jscipy.sparse.linalg.gmres(ggrad, grad)[0]
        step_grad = jnp.clip(step_grad, min_step, max_step)
        
        mu -= step_size*step_grad
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
        
    if ggrad_fn is None:
        ggrad_fn = jacfwdx(grad_fn)
        
    grad = grad_fn_rm(mu_init)
    val, carry = lax.scan(update, init=(mu_init, grad), xs=jnp.arange(0,max_iter,1))

    return val, carry #(mu, grad), (mu, grad)

#%% Joint Newton's Method

@jit 
def JointNewtonsMethod(mu_rm:Array,
                       mu_euc:Array,
                       M:object,
                       grad_fn_rm:Callable[[Tuple[Array, Array]], Array],
                       grad_fn_euc:Callable[[Array], Array],
                       ggrad_fn_rm:Callable[[Tuple[Array, Array]], Array] = None,
                       ggrad_fn_euc:Callable[[Array], Array] = None,
                       step_size_rm:float = 0.1,
                       step_size_euc:float = 0.1,
                       max_iter:int=100,
                       bnds_rm:Tuple[Array, Array]=(None,None),
                       bnds_euc:Tuple[Array, Array]=(None,None),
                       max_step:Array=None
                       )->Tuple[Tuple[Array, Array], Array, Array, Array]:
    
    @jit
    def update(carry:Tuple[jnp.Array, jnp.Array, jnp.Array, jnp.Array], idx:int
               )->Tuple[Tuple[jnp.Array, jnp.Array, jnp.Array, jnp.Array],
                        Tuple[jnp.Array, jnp.Array, jnp.Array, jnp.Array]]:
        
        mu_rm, mu_euc, grad_rm, grad_euc = carry
        
        ggrad_rm = ggrad_fn_rm(mu_rm)
        ggrad_euc = ggrad_fn_euc(mu_euc)
        
        step_grad_rm = jscipy.sparse.linalg.gmres(ggrad_rm, grad_rm)[0]
        step_grad_euc = jscipy.sparse.linalg.gmres(ggrad_euc, grad_euc)[0]
        
        step_grad_rm = jnp.clip(step_grad_rm, min_step, max_step)
        step_grad_euc = jnp.clip(step_grad_euc, min_step, max_step)
        
        mu_rm = M.Exp(mu_rm, -step_size_rm*step_grad_rm)
        mu_rm[0] = jnp.clip(mu_rm[0], lb_rm, ub_rm)
        
        mu_euc -= step_size_euc*step_grad_euc
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
        
    if ggrad_fn_rm is None:
        ggrad_fn_rm = jacfwdx(grad_fn_rm)
        
    if ggrad_fn_euc is None:
        ggrad_fn_euc = jacfwdx(grad_fn_euc)
        
    grad_rm = grad_fn_rm(mu_rm)
    grad_euc = grad_fn_euc(mu_euc)
    val, carry = lax.scan(update, init=(mu_rm, mu_euc, grad_rm, grad_euc), xs=jnp.arange(0,max_iter,1))
    
    return val, carry #(mu_rm, mu_euc, grad_rm, grad_euc), (mu_rm, mu_euc, grad_rm, grad_euc)




