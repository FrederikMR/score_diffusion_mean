#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:26:26 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeoemtry.setup import *

#%% Euler Poincare

def initialize(G:object)->None:
    """ Euler-Poincare geodesic integration """

    def ode_EP(c:tuple[ndarray, ndarray, ndarray],
               y:ndarray
               )->ndarray:
        t,mu,_ = c
        xi = G.invFl(mu)
        dmut = -G.coad(xi,mu)
        
        return dmut

    # reconstruction
    def ode_EPrec(c:tuple[ndarray, ndarray, ndarray],
                  y:tuple[ndarray, ndarray]
                  )->ndarray:
        t,g,_ = c
        mu, = y
        xi = G.invFl(mu)
        dgt = G.dL(g,G.e,G.VtoLA(xi))
        
        return dgt
    
    assert(G.invariance == 'left')
    
    G.EP = lambda mu,_dts=None: integrate(ode_EP,None,mu,None,dts() if _dts is None else _dts)
    
    G.EPrec = lambda g,mus,_dts=None: integrate(ode_EPrec,None,g,None,dts() if _dts is None else _dts,mus)

    ### geodesics
    G.coExpEP = lambda g,mu: G.EPrec(g,G.EP(mu)[1])[1][-1]
    G.ExpEP = lambda g,v: G.coExpEP(g,G.flatV(v))
    G.ExpEPpsi = lambda q,v: G.ExpEP(G.psi(q),G.flatV(v))
    G.coExpEPt = lambda g,mu: G.EPrec(g,G.EP(mu)[1])
    G.ExpEPt = lambda g,v: G.coExpEPt(g,G.flatV(v))
    G.ExpEPpsit = lambda q,v: G.ExpEPt(G.psi(q),G.flatV(v))
    G.DcoExpEP = lambda g,mu: jacrev(G.coExpEP)(g,mu)
    
    return
