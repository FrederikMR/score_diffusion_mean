#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:53:15 2023

@author: frederik
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Logaritmic Map

def initialize(M:object,
               f=None,
               method='BFGS'
               )->None:
    """ numerical Riemannian Logarithm map """

    def loss(x:tuple[ndarray, ndarray],
             v:ndarray,
             y:tuple[ndarray, ndarray]
             )->float:
        
        (x1,chart1) = f(x,v)
        y_chart1 = M.update_coords(y,chart1)
        
        return 1./M.dim*jnp.sum(jnp.square(x1 - y_chart1[0]))
    
    def shoot(x:tuple[ndarray, ndarray],
              y:tuple[ndarray, ndarray],
              v0:ndarray=None
              )->tuple[ndarray, ndarray]:

        if v0 is None:
            v0 = jnp.zeros(M.dim)

        res = minimize(lambda w: (loss(x,w,y),dloss(x,w,y)), v0, method=method, jac=True, options={'disp': False, 'maxiter': 100})

        return (res.x,res.fun)
    
    def dist(x:tuple[ndarray, ndarray],
             y:tuple[ndarray, ndarray]
             )->float:
        
        v = M.Log(x,y)
        
        curve = M.geodesic(x,v[0],dts(T,n_steps))
        
        dt = jnp.diff(curve[0])
        val = vmap(lambda v: M.norm(x,v))(curve[1][:,1])
        
        return jnp.sum(0.5*dt*(val[1:]+val[:-1])) #trapezoidal rule

    if f is None:
        print("using M.Exp for Logarithm")
        f = M.Exp

    dloss = grad(loss,1)

    M.Log = shoot
    M.dist = dist
    
    return
