#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:41:22 2023

@author: frederik
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Riemannian Geodesics

def initialize(M:object) -> None:
    
    def ode_geodesic(c:tuple[ndarray, ndarray, ndarray],y:ndarray)->ndarray:
        t,x,chart = c
        dx2t = -jnp.einsum('ikl,k,l->i',M.Gamma_g((x[0],chart)),x[1],x[1])
        dx1t = x[1] 
        
        return jnp.stack((dx1t,dx2t))
    
    def chart_update_geodesic(xv:ndarray,chart:ndarray,y:ndarray)->tuple[ndarray, ndarray]:
        if M.do_chart_update is None:
            return (xv,chart)
    
        v = xv[1]
        x = (xv[0],chart)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                                jnp.stack((new_x,M.update_vector(x,new_x,new_chart,v))),
                                xv),
                jnp.where(update,
                                new_chart,
                                chart))
    
    def Exp(x:tuple[ndarray, ndarray],
            v:ndarray,
            T:float=T,
            n_steps:int=n_steps
            )->tuple[ndarray, ndarray]:
        curve = M.geodesic(x,v,dts(T,n_steps))
        x = curve[1][-1,0]
        chart = curve[2][-1]
        
        return(x,chart)

    def Expt(x:tuple[ndarray, ndarray],
             v:ndarray,
             T:float=T,
             n_steps:int=n_steps
             )->tuple[ndarray, ndarray]:
        
        curve = M.geodesic(x,v,dts(T,n_steps))
        xs = curve[1][:,0]
        charts = curve[2]
        return(xs,charts)
    
    M.geodesic = jit(lambda x,v,dts: integrate(ode_geodesic,chart_update_geodesic,jnp.stack((x[0],v)),x[1],dts))
    M.Exp = Exp
    M.Expt = Expt