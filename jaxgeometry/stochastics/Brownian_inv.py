#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:36:28 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Brownian Motion with right/left Invariant metric

def initialize(G:object)->None:
    """ Brownian motion with respect to left/right invariant metric """

    def sde_Brownian_inv(c:tuple[ndarray, ndarray, ndarray],
                         y:tuple[ndarray, ndarray]
                         )->tuple[ndarray, ndarray, ndarray, ndarray]:
        t,g,_,sigma = c
        dt,dW = y

        X = jnp.tensordot(G.invpf(g,G.eiLA),sigma,(2,0))
        det = -.5*jnp.tensordot(jnp.diagonal(G.C,0,2).sum(1),X,(0,2))
        sto = jnp.tensordot(X,dW,(2,0))
        
        return (det,sto,X,jnp.zeros_like(sigma))
    
    assert(G.invariance == 'left')

    G.sde_Brownian_inv = sde_Brownian_inv
    G.Brownian_inv = lambda g,dts,dWt,sigma=jnp.eye(G.dim): integrate_sde(G.sde_Brownian_inv,integrator_stratonovich,None,g,None,dts,dWt,sigma)[0:3]

    return