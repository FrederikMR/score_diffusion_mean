#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:25:18 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Stochastic Coadjoint

def initialize(G,Psi=None,r=None):
    """ stochastic coadjoint motion with left/right invariant metric
    see Noise and dissipation on coadjoint orbits arXiv:1601.02249 [math.DS]
    and EulerPoincare.py """

    def sde_stochastic_coadjoint(c:tuple[ndarray, ndarray, ndarray],
                                 y:tuple[ndarray, ndarray]
                                 )->tuple[ndarray, ndarray, ndarray]:
        t,mu,_ = c
        dt,dW = y

        xi = G.invFl(mu)
        det = -G.coad(xi,mu)
        Sigma = G.coad(mu,jax.jacrev(Psi)(mu).transpose((1,0)))
        sto = jnp.tensordot(Sigma,dW,(1,0))
        
        return (det,sto,Sigma)
    
    assert(G.invariance == 'left')

    # Matrix function Psi:LA\rightarrow R^r must be defined beforehand
    # example here from arXiv:1601.02249
    if Psi is None:
        sigmaPsi = jnp.eye(G.dim)
        Psi = lambda mu: jnp.dot(sigmaPsi,mu)
        # r = Psi.shape[0]
        r = G.dim
    assert(Psi is not None and r is not None)
    
    
    G.sde_stochastic_coadjoint = sde_stochastic_coadjoint
    G.stochastic_coadjoint = lambda mu,dts,dWt: integrate_sde(G.sde_stochastic_coadjoint,integrator_stratonovich,None,mu,None,dts,dWt)

    # reconstruction as in Euler-Poincare / Lie-Poisson reconstruction
    if not hasattr(G,'EPrec'):
        from src.group import EulerPoincare
        EulerPoincare.initialize(G)
        
    G.stochastic_coadjointrec = G.EPrec
    
    return