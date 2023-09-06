#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:13:29 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Langevin

def initialize(M:object)->None:

    def sde_Langevin(c:tuple[ndarray, ndarray, ndarray, ndarray, ndarray],
                     y:tuple[ndarray, ndarray]
                     )->tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        
        t,x,chart,l,s = c
        dt,dW = y
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])-l*dq((x[0],chart),x[1])

        X = jnp.stack((jnp.zeros((M.dim,M.dim)),s*jnp.eye(M.dim)))
        det = jnp.stack((dqt,dpt))
        sto = jnp.tensordot(X,dW,(1,0))
        return (det,sto,X,jnp.zeros_like(l),jnp.zeros_like(s))

    def chart_update_Langevin(xp:ndarray,
                              chart:ndarray,
                              *cy
                              )->tuple[ndarray, ndarray, ...]:
        
        if M.do_chart_update is None:
            return (xp,chart,*cy)
    
        p = xp[1]
        x = (xp[0],chart)
    
        update = M.do_chart_update(x)
        new_chart = M.centered_chart(M.F(x))
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                            jnp.stack((new_x,M.update_covector(x,new_x,new_chart,p))),
                            xp),
                jnp.where(update,
                            new_chart,
                            chart),
                *cy)
    
    dq = jit(grad(M.H,argnums=1))
    dp = jit(lambda q,p: -gradx(M.H)(q,p))

    M.Langevin_qp = lambda q,p,l,s,dts,dWt: integrate_sde(sde_Langevin,integrator_ito,chart_update_Langevin,jnp.stack((q[0],p)),q[1],dts,dWt,l,s)

    M.Langevin = lambda q,p,l,s,dts,dWt: M.Langevin_qp(q,p,l,s,dts,dWt)[0:3]
    
    return