#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:40:00 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Brownian process with respect to left/right invariant metric

def initialize(G:object)->None:
    """ Brownian motion with respect to left/right invariant metric """

    assert(G.invariance == 'left')

    def sde_Brownian_process(c:tuple[ndarray, ndarray, ndarray],
                             y:tuple[ndarray, ndarray]
                             )->tuple[ndarray, ndarray, ndarray, ndarray]:
        t,g,_,sigma = c
        dt,dW = y

        X = jnp.tensordot(G.invpf(g,G.eiLA),sigma,(2,0))
        det = jnp.zeros_like(g)
        sto = jnp.tensordot(X,dW,(2,0))
        
        return (det,sto,X,0.)

    G.sde_Brownian_process = sde_Brownian_process
    G.Brownian_process = lambda g,dts,dWt,sigma=jnp.eye(G.dim): integrate_sde(G.sde_Brownian_process,integrator_stratonovich,None,g,None,dts,dWt,sigma)[0:3]
    
    return

