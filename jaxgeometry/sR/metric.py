#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:11:16 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% metric

def initialize(M:object)->None:
    """ add SR structure to manifold """
    """ currently assumes distribution and that ambient Riemannian manifold is Euclidean """

    if hasattr(M, 'D'):
        M.a = lambda x: jnp.dot(M.D(x),M.D(x).T)
    else:
        raise ValueError('no metric or cometric defined on manifold')

    ##### sharp map:
    M.sharp = lambda x,p: jnp.tensordot(M.a(x),p,(1,0))

    ##### Hamiltonian
    M.H = lambda x,p: .5*jnp.sum(jnp.dot(p,M.sharp(x,p))**2)
    
    return
