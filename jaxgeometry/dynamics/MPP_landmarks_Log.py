#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:53:10 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% MPP Landmarks Log

###############################################################
# Most probable paths for landmarks via development - BVP     # 
###############################################################
def initialize(M:object,
               method:str='BFGS'
               )->None:

    def loss(x:ndarray,
             lambd:ndarray,
             y:ndarray,
             qps:ndarray,
             _dts:ndarray
             )->ndarray:
        
        (_,xs,_,charts) = M.MPP_landmarks(x,lambd,qps,_dts)
        (x1,chart1) = (xs[-1],charts[-1])
        y_chart1 = M.update_coords(y,chart1)
        
        return 1./M.dim*jnp.sum(jnp.square(x1 - y_chart1[0]))

    def shoot(x:ndarray,
              y:ndarray,
              qps:ndarray,
              _dts:ndarray,
              lambd0:ndarray=None
              )->tuple[ndarray, ndarray]:        

        if lambd0 is None:
            lambd0 = jnp.zeros(M.dim)

        res = minimize(jax.value_and_grad(lambda w: loss(x,w,y,qps,_dts)), lambd0, method=method, jac=True, options={'disp': False, 'maxiter': 100})

        return (res.x,res.fun)

    M.Log_MPP_landmarks = shoot
    
    return