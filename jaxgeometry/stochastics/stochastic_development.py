#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:29:29 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Stochastic Development

def initialize(M:object)->None:
    """ development and stochastic development from R^d to M """

    # Deterministic development
    def ode_development(c:tuple[ndarray, ndarray, ndarray],
                        y:tuple[ndarray, ndarray]):
        t,u,chart = c
        dgamma, = y

        u = (u,chart)
        nu = u[0][M.dim:].reshape((M.dim,-1))
        m = nu.shape[1]

        det = jnp.tensordot(M.Horizontal(u)[:,0:m],dgamma,(1,0))
    
        return det

    # Stochastic development
    def sde_development(c:tuple[ndarray, ndarray, ndarray],
                        y:tuple[ndarray, ndarray]
                        )->tuple[ndarray, ndarray, dnarray]:
        
        t,u,chart = c
        dt,dW = y

        u = (u,chart)
        nu = u[0][M.dim:].reshape((M.dim,-1))
        m = nu.shape[1]

        sto = jnp.tensordot(M.Horizontal(u)[:,0:m],dW,(1,0))
    
        return (jnp.zeros_like(sto), sto, M.Horizontal(u)[:,0:m])
    
    M.development = jit(lambda u,dgamma,dts: integrate(ode_development,M.chart_update_FM,u[0],u[1],dts,dgamma))

    M.sde_development = sde_development
    M.stochastic_development = jit(lambda u,dts,dWs: integrate_sde(sde_development,integrator_stratonovich,M.chart_update_FM,u[0],u[1],dts,dWs))
    
    return