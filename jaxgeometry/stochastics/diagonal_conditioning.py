#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:44:32 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Diagonal Conditioning

def initialize(M:object,
               sde_product:Callable[[tuple[ndarray, ndarray, ndarray, ...], ndarray], 
                                    tuple[ndarray, ndarray, ndarray, ...]],
               chart_update_product:Callable[[tuple[ndarray, ndarray], ndarray, ...], 
                                    tuple[ndarray, ndarray, ...]],
               integrator:Callable[[ndarray], ndarray]=integrator_ito,
               T:float=1
               )->None:
    """ diagonally conditioned product diffusions """

    def sde_diagonal(c:tuple[ndarray, ndarray, ndarray, ...],
                     y:tuple[ndarray, ndarray]
                     )->tuple[ndarray, ndarray, ndarray, ndarray, ...]:
        
        if M.do_chart_update is None:
            t,x,chart,T,*cy = c
        else:
            t,x,chart,T,ref_chart,*cy = c
        dt,dW = y
        
        (det,sto,X,*dcy) = sde_product((t,x,chart,*cy),y)

        if M.do_chart_update is None:
            xref = x
        else:
            xref = vmap(lambda x,chart: M.update_coords((x,chart),ref_chart)[0],0)(x,chart)
        m = jnp.mean(xref,0) # mean
        href = cond(t<T-dt/2,
                 lambda _: (m-xref)/(T-t),
                 lambda _: jnp.zeros_like(det),
                 None)
        if M.do_chart_update is None:
            h = href
        else:
            h = vmap(lambda xref,x,chart,h: M.update_vector((xref,ref_chart),x,chart,h),0)(xref,x,chart,href)
        
        # jnp.tensordot(X,h,(2,1))
        if M.do_chart_update is None:
            return (det+h,sto,X,0.,*dcy)
        else:
            return (det+h,sto,X,0.,jnp.zeros_like(ref_chart),*dcy)

    def chart_update_diagonal(x:ndarray,
                              chart:ndarray,
                              *ys
                              )->tuple[ndarray, ndarray, ndarray, float, ...]:
        if M.do_chart_update is None:
            return (x,chart,*ys)

        (ref_chart,T,*_ys) = ys

        (new_x,new_chart,*new_ys) = chart_update_product(x,chart,*_ys)
        return (new_x,new_chart,ref_chart,T,*new_ys)
    
    M.sde_diagonal = sde_diagonal
    M.chart_update_diagonal = chart_update_product
    if M.do_chart_update is None:
        M.diagonal = jit(lambda x,dts,dWt: integrate_sde(sde_diagonal,integrator,M.chart_update_diagonal,x[0],x[1],dts,dWt,jnp.sum(dts))[0:3])
    else:
        M.diagonal = jit(lambda x,dts,dWt,ref_chart,*ys: integrate_sde(sde_diagonal,integrator,chart_update_diagonal,x[0],x[1],dts,dWt,jnp.sum(dts),ref_chart,*ys)[0:3])

    return
