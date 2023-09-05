#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:32:18 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.seutup import *

#%% MPP Kunita Log

###############################################################
# Most probable paths for Kunita flows - BVP                  # 
###############################################################

def initialize(M:object,
               N:object,
               method:str='BFGS'
               )->None:

    def loss(x:ndarray,
             v:ndarray,
             y:ndarray,
             qps:ndarray,
             dqps:ndarray,
             _dts:ndarray
             )->ndarray:
        
        (_,xx1,charts) = M.MPP_AC(x,v,qps,dqps,_dts)
        (x1,chart1) = (xx1[-1,0],charts[-1])
        y_chart1 = M.update_coords(y,chart1)
        
        return 1./N.dim*jnp.sum(jnp.square(x1 - y_chart1[0]))

    def shoot(x:ndarray,
              y:ndarray,
              qps:ndarray,
              dqps:ndarray,
              _dts:ndarray,
              v0:ndarray=None
              )->tuple[ndarray, ndarray]:        

        if v0 is None:
            v0 = jnp.zeros(N.dim)

        #res = minimize(jax.value_and_grad(lambda w: loss(x,w,y,qps,dqps,_dts)), v0, method=method, jac=True, options={'disp': False, 'maxiter': 100})
        res = minimize(lambda w: (loss(x,w,y,qps,dqps,_dts),dloss(x,w,y,qps,dqps,_dts)), 
                       v0, method=method, jac=True, options={'disp': False, 'maxiter': 100})
    #     res = minimize(lambda w: loss(x,w,y,qps,dqps,_dts), v0, method=method, jac=False, options={'disp': False, 'maxiter': 100})

    #     print(res)

        return (res.x,res.fun)

    dloss = lambda x,v,y,qps,dqps,_dts: approx_fprime(v,lambda v: loss(x,v,y,qps,dqps,_dts),1e-4)
    M.Log_MPP_AC = shoot
    
    return