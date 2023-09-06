#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:28:53 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Brownian coords

def initialize(M:object)->None:
    """ Brownian motion in coordinates """

    def sde_Brownian_coords(c:tuple[ndarray, ndarray, ndarray],
                            y:tuple[ndarray, ndarray]
                            )->tuple[ndarray, ndarray, ndarray, float]:
        
        t,x,chart,s = c
        dt,dW = y

        gsharpx = M.gsharp((x,chart))
        X = s*jnp.linalg.cholesky(gsharpx)
        det = -.5*(s**2)*jnp.einsum('kl,ikl->i',gsharpx,M.Gamma_g((x,chart)))
        sto = jnp.tensordot(X,dW,(1,0))
        
        return (det,sto,X,0.)
    
    def chart_update_Brownian_coords(x:tuple[ndarray, ndarray],
                                     chart:ndarray,
                                     *ys
                                     )->tuple[ndarray, ndarray, ...]:
        
        if M.do_chart_update is None:
            return (x,chart,*ys)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart((x,chart))
        new_x = M.update_coords((x,chart),new_chart)[0]

        return (jnp.where(update,
                                new_x,
                                x),
                jnp.where(update,
                                new_chart,
                                chart),
                *ys)
    
    M.sde_Brownian_coords = sde_Brownian_coords
    M.chart_update_Brownian_coords = chart_update_Brownian_coords
    M.Brownian_coords = jit(lambda x,dts,dWs,stdCov=1.: integrate_sde(sde_Brownian_coords,
                                                                      integrator_ito,
                                                                      chart_update_Brownian_coords,
                                                                      x[0],x[1],dts,dWs,stdCov)[0:3])
    
    return
