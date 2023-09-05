#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:30:30 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% MPP landmarks

###############################################################
# Most probable paths for landmarks via development           #
###############################################################
def initialize(M:object,
               sigmas:ndarray,
               dsigmas:ndarray,
               a:Callable[[ndarray], ndarray]):
    """ Most probable paths for Kunita flows                 """
    """ M: shape manifold, a: flow field                     """
    
    def ode_MPP_landmarks(c:tuple[ndarray, ndarray, ndarray],
                          y:tuple[ndarray, ndarray]
                          )->tuple[ndarray, ndarray]:
        
        t,xlambd,chart = c
        qp, = y
        x = xlambd[0].reshape((M.N,M.m))  # points
        lambd = xlambd[1].reshape((M.N,M.m))
        
        sigmasx = sigmas(x)
        dsigmasx = dsigmas(x)
        c = jnp.einsum('ri,rai->a',lambd,sigmasx)

        dx = a(x,qp)+jnp.einsum('a,rak->rk',c,sigmasx)
        #dlambd = -jnp.einsum('ri,a,rairk->rk',lambd,c,jacrev(sigmas)(x))-jnp.einsum('ri,rirk->rk',lambd,jacrev(a)(x,qp))
        dlambd = -jnp.einsum('ri,a,raik->rk',lambd,c,dsigmasx)-jnp.einsum('ri,rirk->rk',lambd,jacrev(a)(x,qp))
        
        return jnp.stack((dx.flatten(),dlambd.flatten()))

    def chart_update_MPP_landmarks(xlambd:tuple[ndarray, ndarray],
                                   chart:ndarray,
                                   y:ndarray
                                   )->tuple[ndarray, ndarray]:
        
        if M.do_chart_update is None:
            return (xlambd,chart)
    
        lambd = xlambd[1].reshape((M.N,M.m))
        x = (xlambd[0],chart)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                                jnp.stack((new_x,M.update_covector(x,new_x,new_chart,lambd))),
                                xlambd),
                jnp.where(update,
                                new_chart,
                                chart))
    
    def MPP_landmarks(x:tuple[ndarray, ndarray],
                      lambd:ndarray,
                      qps:ndarray,
                      dts:ndarray
                      )->tuple[ndarray, ndarray, ndarray]:
        
        (ts,xlambds,charts) = integrate(ode_MPP_landmarks,chart_update_MPP_landmarks,jnp.stack((x[0],lambd)),x[1],dts,qps)
        
        return (ts,xlambds[:,0],xlambds[:,1],charts)
    
    M.MPP_landmarks = MPP_landmarks

    return