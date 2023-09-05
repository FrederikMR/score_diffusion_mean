#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:08:44 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Energy

def initialize(G:object)->None:
    """ group Lagrangian and Hamiltonian from invariant metric """

    def Lagrangian(g:ndarray,
                   vg:ndarray
                   )->ndarray:
        
        """ Lagrangian """
        
        return .5*G.gG(g,vg,vg)
    
    def Lagrangianpsi(q:ndarray,
                      v:ndarray
                      )->ndarray:
        
        """ Lagrangian using psi map """
        
        return .5*G.gpsi(q,v,v)
    
    def l(hatxi:ndarray)->ndarray:
        
        """ LA restricted Lagrangian """
        
        return 0.5*G.gV(hatxi,hatxi)

    def Hpsi(q:ndarray,
             p:ndarray
             )->ndarray:
        
        """ Hamiltonian using psi map """
        
        return .5*G.cogpsi(q,p,p)
    
    def Hminus(mu:ndarray)->ndarray:
        
        """ LA^* restricted Hamiltonian """
        
        return .5*G.cogV(mu,mu)

    def HL(q:ndarray,
           p:ndarray
           )->ndarray:
        
        """ Legendre transformation. The above Lagrangian is hyperregular """
        
        (q,v) = invFLpsi(q,p)
        
        return jnp.dot(p,v)-L(q,v)
    
    def hl(mu:ndarray)->ndarray:
        
        hatxi = invFl(mu)
        
        return jnp.dot(mu,hatxi)-l(hatxi)
    
    G.Lagrangian = Lagrangian
    
    G.Lagrangianpsi = Lagrangianpsi
    G.dLagrangianpsidq = jax.grad(G.Lagrangianpsi)
    G.dLagrangianpsidv = jax.grad(G.Lagrangianpsi)
    
    G.l = l
    G.dldhatxi = jax.grad(G.l)
    
    G.Hpsi = Hpsi
    
    G.Hminus = Hminus
    G.dHminusdmu = jax.grad(G.Hminus)
    
    G.FLpsi = lambda q,v: (q,G.dLagrangianpsidv(q,v))
    G.invFLpsi = lambda q,p: (q,G.cogpsi(q,p))
    
    G.HL = HL
    G.Fl = lambda hatxi: G.dldhatxi(hatxi)
    G.invFl = lambda mu: G.cogV(mu)
    
    G.hl = hl

    # default Hamiltonian
    G.H = lambda q,p: G.Hpsi(q[0],p) if type(q) == type(()) else G.Hpsi(q,p)
    
    return

# A.set_value(np.diag([3,2,1]))
# print(FLpsif(q0,v0))
# print(invFLpsif(q0,p0))
# (flq0,flv0)=FLpsif(q0,v0)
# print(q0,v0)
# print(invFLpsif(flq0,flv0))