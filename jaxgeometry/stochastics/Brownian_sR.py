#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:42:38 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setupt import *

#%% sub-Riemannian Brownian Motion

def initialize(M):
    """ sub-Riemannian Brownian motion """

    def sde_Brownian_sR(c:tuple[ndarray, ndarray, ndarray],
                        y:tuple[ndarray, ndarray]
                        )->tuple[ndarray, ndarray, ndarray]:
        t,x,chart = c
        dt,dW = y

        D = M.D((x,chart))
        # D0 = \sum_{i=1}^m div_\mu(X_i) X_i) - not implemented yet
        det = jnp.zeros_like(x) # Y^k(x)=X_0^k(x)+(1/2)\sum_{i=1}^m \langle \nabla X_i^k(x),X_i(x)\rangle
        sto = jnp.tensordot(D,dW,(1,0))
        
        return (det,sto,D)
    
    def chart_update_Brownian_sR(x:ndarray,
                                 chart:ndarray,*ys
                                 )->tuple[ndarray, ndarray]:
        if M.do_chart_update is None:
            return (x,chart,*ys)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart((x,chart))
        new_x = M.update_coords((x,chart),new_chart)[0]

        return (jnp.where(update,
                                new_x,
                                x),
                jnp.where(update,
                                new_chart,
                                chart),*ys)
    
    M.sde_Brownian_sR = sde_Brownian_sR
    M.chart_update_Brownian_sR = chart_update_Brownian_sR
    M.Brownian_sR = jit(lambda x,dts,dWs: integrate_sde(sde_Brownian_sR,integrator_ito,chart_update_Brownian_sR,x[0],x[1],dts,dWs))

    return