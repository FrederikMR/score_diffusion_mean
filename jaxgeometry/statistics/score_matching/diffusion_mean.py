#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:37:20 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *
from jaxgeometry.optimization.JAXOptimization import JointJaxOpt

#%% Diffusion Mean Estimation

def initialize(M:object,
               s1_model:Callable[[ndarray, ndarray, ndarray], ndarray],
               s2_model:Callable[[ndarray, ndarray, ndarray], ndarray],
               method="JAX"
               )->None:
    
    @jit
    def gradt_loss(X_obs:Tuple[ndarray, ndarray],y:Tuple[ndarray, ndarray],t:ndarray)->ndarray:
        
        s1 = vmap(lambda x, chart: s1_model((x,chart),y,t))(X_obs[0], X_obs[1])
        s2 = vmap(lambda x, chart: s2_model((x,chart),y,t))(X_obs[0], X_obs[1])
        
        div = vmap(lambda s1, s2, x, chart: jnp.trace(s2)+.5*jnp.dot(s1,jacfwdx(M.logAbsDet)((x,chart)).squeeze()))(s1,
                                                                                                s2,
                                                                                                X_obs[0], 
                                                                                                X_obs[1])
        
        return -0.5*jnp.mean(vmap(lambda s1, div: jnp.dot(s1, s1)+div)(s1, div), axis=0)
    
    @jit
    def gradx_loss(X_obs:Tuple[ndarray, ndarray], y:Tuple[ndarray,ndarray],t:ndarray)->ndarray:
        
        s1 = vmap(lambda x,chart: s1_model((x,chart),y,t))(X_obs[0], X_obs[1])
        
        gradx = -jnp.mean(s1, axis=0)
        
        return gradx
    
    M.sm_dmxt = lambda X_obs, x0, t: JointJaxOpt(x0,
                                                 jnp.array(t),
                                                 M,
                                                 grad_fn_rm = lambda y,t: gradx_loss(X_obs, y, t),
                                                 grad_fn_euc = lambda y,t: gradt_loss(X_obs, y, t),
                                                 max_iter=1000,
                                                 bnds_euc=(0.0,None),
                                                 )
    M.sm_dmx = lambda X_obs, x0, t: RMJaxOpt(x0,
                                             M,
                                             grad_fn=lambda y: gradx_loss(X_obs, y, t),
                                             max_iter=1000,
                                             )
    M.sm_dmt = lambda X_obs, x0, t: JaxOpt(t,
                                           M,
                                           grad_fn = lambda t: gradt_loss(X_obs, x0, t),
                                           max_iter=1000,
                                           bnds=(0.0,None),
                                           )
    
    return