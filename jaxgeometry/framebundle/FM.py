#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:39:38 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Frame Bunddle

def initialize(M:object)->None:
    """ Frame Bundle geometry """

    def chart_update_FM(u:ndarray,
                        chart:ndarray,
                        *args
                        )->tuple[ndarray, ndarray]:
        
        if M.do_chart_update != True:
            return (u,chart)
        
        x = (u[0:d],chart)
        nu = u[d:].reshape((d,-1))

        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
        
        return (jnp.where(update,
                                jnp.concatenate((new_x,M.update_vector(x,new_x,new_chart,nu).flatten())),
                                u),
                jnp.where(update,
                                new_chart,
                                chart)) 

    #### Bases shifts, see e.g. Sommer Entropy 2016 sec 2.3
    # D denotes frame adapted to the horizontal distribution
    def to_D(u:ndarray,
             w:ndarray
             )->ndarray:
        
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        wx = w[0:d]
        wnu = w[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = jnp.tensordot(M.Gamma_g(x),nu,(2,0)).swapaxes(1,2)
        Dwx = wx
        Dwnu = jnp.tensordot(Gammanu,wx,(2,0))+wnu

        return jnp.concatenate((Dwx,Dwnu.flatten()))
    
    def from_D(u:ndarray,
               Dw:ndarray
               )->ndarray:
        
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        Dwx = Dw[0:d]
        Dwnu = Dw[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = jnp.tensordot(M.Gamma_g(x),nu,(2,0)).swapaxes(1,2)
        wx = Dwx
        wnu = -jnp.tensordot(Gammanu,Dwx,(2,0))+Dwnu

        return jnp.concatenate((wx,wnu.flatten())) 
        # corresponding dual space shifts
    def to_Dstar(u:ndarray,
                 p:ndarray
                 )->ndarray:
        
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        px = p[0:d]
        pnu = p[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = jnp.tensordot(M.Gamma_g(x),nu,(2,0)).swapaxes(1,2)
        Dpx = px-jnp.tensordot(Gammanu,pnu,((0,1),(0,1)))
        Dpnu = pnu

        return jnp.concatenate((Dpx,Dpnu.flatten()))
    
    def from_Dstar(u:ndarray,
                   Dp:ndarray
                   )->ndarray:
        
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        Dpx = Dp[0:d]
        Dpnu = Dp[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = jnp.tensordot(M.Gamma_g(x),nu,(2,0)).swapaxes(1,2)
        px = Dpx+jnp.tensordot(Gammanu,Dpnu,((0,1),(0,1)))
        pnu = Dpnu

        return jnp.concatenate((px,pnu.flatten()))
    
    ##### Horizontal vector fields:
    def Horizontal(u:ndarray)->ndarray:
        
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
    
        # Contribution from the coordinate basis for x: 
        dx = nu
        # Contribution from the basis for Xa:
        Gammahgammaj = jnp.einsum('hji,ig->hgj',M.Gamma_g(x),nu) # same as Gammanu above
        dnu = -jnp.einsum('hgj,ji->hgi',Gammahgammaj,nu)

        return jnp.concatenate([dx,dnu.reshape((-1,nu.shape[1]))],axis=0)
    
    d  = M.dim
    
    M.chart_update_FM = chart_update_FM
    
    M.to_D = to_D
    M.from_D = from_D
    M.to_Dstar = to_Dstar
    M.from_Dstar = from_Dstar
    
    M.Horizontal = Horizontal
    
    return