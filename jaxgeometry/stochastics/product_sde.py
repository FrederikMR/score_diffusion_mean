#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:18:02 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Product Diffusion Process

def initialize(M:object,
               sde:Callable[[tuple[ndarray, ndarray, ndarray, ...], tuple[ndarray, ndarray]],
                            tuple[ndarray, ndrray, dnarray, ...]],
               chart_update:Callable,
               integrator:Callable=integrator_ito):
    """ product diffusions """

    def sde_product(c:tuple[ndarray, ndarray, ndarray, ...],
                    y:tuple[ndarray, ndarray]
                    )->tuple[ndarray, ndarray, ndarray, ...]:
        t,x,chart,*cy = c
        dt,dW = y
        
        (det,sto,X,*dcy) = vmap(lambda x,chart,dW,*_cy: sde((t,x,chart,*_cy),(dt,dW)),0)(x,chart,dW,*cy)

        return (det,sto,X,*dcy)

    chart_update_product = vmap(chart_update)

    product = jit(lambda x,dts,dWs,*cy: integrate_sde(sde_product,integrator,chart_update_product,x[0],x[1],dts,dWs,*cy))

    return (product,sde_product,chart_update_product)

# for initializing parameters
def tile(x:ndarray,N:int):
    
    try:
        return jnp.tile(x,(N,)+(1,)*x.ndim)
    except AttributeError:
        try:
            return jnp.tile(x,N)
        except TypeError:
            return tuple([tile(y,N) for y in x])