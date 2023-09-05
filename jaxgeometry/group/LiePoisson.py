#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:39:11 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Lie Poisson

def initialize(G:object)->None:
    """ Lie-Poisson geodesic integration """

    def ode_LP(c:tuple[ndarray, ndarray, ndarray],
               y:tuple[ndarray, ndarray]
               )->ndarray:
        
        t,mu,_ = c
        dmut = G.coad(G.dHminusdmu(mu),mu)
        
        return dmut

    # reconstruction
    def ode_LPrec(c:tuple[ndarray, ndarray, ndarray],
                  y:tuple[ndarray, ndarray]
                  )->ndarray:
        t,g,_ = c
        mu, = y
        dgt = G.dL(g,G.e,G.VtoLA(G.dHminusdmu(mu)))
        return dgt
    
    assert(G.invariance == 'left')
    
    G.LP = lambda mu,_dts=None: integrate(ode_LP,None,mu,None,dts() if _dts is None else _dts)
    G.LPrec = lambda g,mus,_dts=None: integrate(ode_LPrec,None,g,None,dts() if _dts is None else _dts,mus)

    ### geodesics
    G.coExpLP = lambda g,mu: G.LPrec(g,G.LP(mu)[1])[1][-1]
    G.ExpLP = lambda g,v: G.coExpLP(g,G.flatV(v))
    G.coExpLPt = lambda g,mu: G.LPrec(g,G.LP(mu)[1])
    G.ExpLPt = lambda g,v: G.coExpLPt(g,G.flatV(v))
    G.DcoExpLP = lambda g,mu: jax.jacrev(G.coExp)(g,mu)
    
    return