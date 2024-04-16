#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:37:20 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *
from jaxgeometry.optimization.JAXOptimization import JointJaxOpt, RMJaxOpt, JaxOpt
from jaxgeometry.optimization.GradientDescent import JointGradientDescent, RMGradientDescent, GradientDescent

#%% Diffusion Mean Estimation

def diffusion_mean(M:object,
                   s1_model:Callable[[Array, Array, Array], Array],
                   s2_model:Callable[[Array, Array, Array], Array],
                   method:str="JAX",
                   min_t:float = 1e-3,
                   max_t:float = 1.0,
                   )->None:
    
    """
    @jit
    def gradt_loss(X_obs:Tuple[Array, Array],y:Tuple[Array, Array],t:Array)->Array:
        
        s1 = vmap(lambda x, chart: s1_model((x,chart),y,t))(X_obs[0], X_obs[1])
        s2 = vmap(lambda x, chart: s2_model((x,chart),y,t))(X_obs[0], X_obs[1])
        
            div = vmap(lambda s1, s2, x, chart: jnp.trace(s2)+.5*jnp.dot(s1,jacfwdx(M.logAbsDet)((x,chart)).squeeze()))(s1,
                                                                                                    s2,
                                                                                                    X_obs[0], 
                                                                                                    X_obs[1])
        
        return -0.5*jnp.mean(vmap(lambda s1, div: jnp.dot(s1, s1)+div)(s1, div), axis=0)
    
    @jit
    def gradx_loss(X_obs:Tuple[Array, Array], y:Tuple[Array,Array],t:Array)->Array:
        
        s1 = vmap(lambda x,chart: s1_model((x,chart),y,t))(X_obs[0], X_obs[1])
        
        gradx = -jnp.mean(s1, axis=0)
        
        return gradx
    """
    
    @jit
    def gradt_loss(X_obs:Tuple[Array, Array],y:Tuple[Array, Array],t:Array)->Array:
        
        s2 = vmap(lambda x,chart: s2_model((x,chart),y,t))(X_obs[0], X_obs[1])
        
        return -jnp.mean(s2, axis=0)
    
    @jit
    def gradx_loss(X_obs:Tuple[Array, Array], y:Tuple[Array,Array],t:Array)->Array:
        
        s1 = vmap(lambda x,chart: s1_model((x,chart),y,t))(X_obs[0], X_obs[1])
        
        gradx = -jnp.mean(s1, axis=0)
        
        return gradx
    
    if method == "JAX":
        M.sm_dmxt = lambda X_obs, x0, t,step_size=0.01, max_iter=1000: JointJaxOpt(x0,
                                                     jnp.array(t),
                                                     M,
                                                     grad_fn_rm = lambda y,t: gradx_loss(X_obs, y, t),
                                                     grad_fn_euc = lambda y,t: gradt_loss(X_obs, y, t),
                                                     max_iter=max_iter,
                                                     lr_rate = step_size,
                                                     bnds_euc=(min_t,max_t),
                                                     )[1]
        
        M.sm_dmx = lambda X_obs, x0, t, step_size=0.01, max_iter=1000: RMJaxOpt(x0,
                                                 M,
                                                 grad_fn=lambda y: gradx_loss(X_obs, y, t),
                                                 max_iter=max_iter,
                                                 lr_rate = step_size,
                                                 )[1]
        M.sm_dmt = lambda X_obs, x0, t, step_size=0.01, max_iter=1000: JaxOpt(t,
                                               M,
                                               grad_fn = lambda t: gradt_loss(X_obs, x0, t),
                                               max_iter=max_iter,
                                               lr_rate = step_size,
                                               bnds=(min_t,max_t),
                                               )[1]
    elif method == "Gradient":
        
        M.sm_dmxt = lambda X_obs, x0, t, step_size=0.1, max_iter=1000: JointGradientDescent(x0,
                                                                                            jnp.array(t),
                                                                                            M,
                                                                                            grad_fn_rm = lambda y,t: gradx_loss(X_obs, y, t),
                                                                                            grad_fn_euc = lambda y,t: gradt_loss(X_obs, y, t),
                                                                                            step_size_rm=step_size,
                                                                                            step_size_euc=step_size,
                                                                                            max_iter = max_iter,
                                                                                            bnds_euc = (min_t,max_t),
                                                                                            )[1]
        
        M.sm_dmx = lambda X_obs, x0, t, step_size=0.1, max_iter=1000: RMGradientDescent(x0,
                                                                                        M,
                                                                                        grad_fn = lambda y: gradx_loss(X_obs, y, t),
                                                                                        step_size=step_size,
                                                                                        max_iter = max_iter
                                                                                        )[1]
        
        M.sm_dmt = lambda X_obs, y, t, step_size=0.1, max_iter=1000: GradientDescent(t,
                                                                                     grad_fn = lambda t: gradt_loss(X_obs, y, t),
                                                                                     step_size=step_size,
                                                                                     max_iter = max_iter,
                                                                                     bnds=(min_t,max_t)
                                                                                     )[1]
    
    
    
    return