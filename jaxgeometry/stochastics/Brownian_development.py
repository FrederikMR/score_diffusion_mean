#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:34:49 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Brownian Development

def initialize(M:object)->None:
    """ Brownian motion from stochastic development """

    def Brownian_development(x:tuple[ndarray, ndarray],
                             dts:ndarray,
                             dWs:ndarray
                             )->tuple[ndarray, ndarray, ndarray]:
        # amend x with orthogonal basis to get initial frame bundle element
        gsharpx = M.gsharp(x)
        nu = jnp.linalg.cholesky(gsharpx)
        u = (jnp.concatenate((x[0],nu.flatten())),x[1])
        
        (ts,us,charts) = M.stochastic_development(u,dts,dWs)
        
        return (ts,us[:,0:M.dim],charts)
    
    M.Brownian_development = Brownian_development
    
    return