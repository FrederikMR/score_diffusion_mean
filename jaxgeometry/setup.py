#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:36:41 2023

@author: frederik
"""

#%% Sources

#%% Modules

#Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#JAX
from jax.numpy import ndarray
import jax.numpy as jnp
from jax.lax import stop_gradient, scan, cond
from jax import vmap, grad, jacfwd, jacrev, random, jit

#JAX Optimization
from jax.example_libraries import optimizers

#JAX scipy
import jax.scipy as jscipy

#numpy
import numpy as np

#scipy
from scipy.optimize import minimize,fmin_bfgs,fmin_cg, approx_fprime

#sklearn
from sklearn.decomposition import PCA

#Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#functools
from functools import partial

#typing
from typing import Callable

#JAXGeometry


#%% Default Parameters

# default integration times and time steps
T = 1.
n_steps = 100
seed = 2712 #Old value 42

# Integrator variables:
default_method = 'euler'
#default_method = 'rk4'

global key
key = random.PRNGKey(seed)

#%% Automatic Differentiation

def gradx(f:Callable[[tuple[ndarray, ndarray], ...], ndarray])->Callable[[tuple[ndarray, ndarray], ...], ndarray]:
    """ jax.grad but only for the x variable of a function taking coordinates and chart """
    def fxchart(x:ndarray,chart:ndarray,*args,**kwargs)->ndarray:
        return f((x,chart),*args,**kwargs)
    def gradf(x:tuple[ndarray, ndarray],*args,**kwargs)->ndarray:
        return grad(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return gradf

def jacfwdx(f:Callable[[tuple[ndarray, ndarray], ...], ndarray])->Callable[[tuple[ndarray, ndarray], ...], ndarray]:
    """jax.jacfwd but only for the x variable of a function taking coordinates and chart"""
    def fxchart(x:ndarray,chart:ndarray,*args,**kwargs)->ndarray:
        return f((x,chart),*args,**kwargs)
    def jacf(x:tuple[ndarray, ndarray],*args,**kwargs)->ndarray:
        return jacfwd(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return jacf

def jacrevx(f:Callable[[tuple[ndarray, ndarray], ...], ndarray]):
    """jax.jacrev but only for the x variable of a function taking coordinates and chart"""
    def fxchart(x:ndarray,chart:ndarray,*args,**kwargs):
        return f((x,chart),*args,**kwargs)
    def jacf(x:tuple[ndarray, ndarray],*args,**kwargs)->ndarray:
        return jacrev(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return jacf

def hessianx(f:Callable[[tuple[ndarray, ndarray], ...], ndarray])->ndarray:
    """hessian only for the x variable of a function taking coordinates and chart"""
    return jacfwdx(jacrevx(f))

def straight_through(f:Callable[[tuple[ndarray, ndarray], ...], ndarray],
                     x:tuple[ndarray, ndarray],
                     *ys)->tuple[ndarray, ndarray]:
    """
    evaluation with pass through derivatives
    Create an exactly-zero expression with Sterbenz lemma that has
    an exactly-one gradient.
    """
    if type(x) == type(()):
        #zeros = tuple([xi - stop_gradient(xi) for xi in x])
        fx = stop_gradient(f(x,*ys))
        return tuple([fxi - stop_gradient(fxi) for fxi in fx])
    else:
        return x-stop_gradient(x)
        #zero = x - stop_gradient(x)
        #return zeros + stop_gradient(f(x,*ys))

#%% Integration

def dts(T:int=T,n_steps:int=n_steps)->ndarray:
    """time increments, deterministic"""
    return jnp.array([T/n_steps]*n_steps)

def dWs(d:int,_dts:ndarray=None,num:int=1)->ndarray:
    """
    standard noise realisations
    time increments, stochastic
    """
    global key
    keys = random.split(key,num=num+1)
    key = keys[0]
    subkeys = keys[1:]
    if _dts == None:
        _dts = dts()
    if num == 1:
        return jnp.sqrt(_dts)[:,None]*random.normal(subkeys[0],(_dts.shape[0],d))
    else:
        return vmap(lambda subkey: jnp.sqrt(_dts)[:,None]*random.normal(subkey,(_dts.shape[0],d)))(subkeys)    

def integrator(ode_f:Callable[tuple[ndarray, ndarray, ndarray], ndarray],
               chart_update:bool=None,
               method:str=default_method)->Callable[[tuple[ndarray, ndarray, ndarray], tuple[ndarray, ndarray]], ndarray]:
    """
    Integrator (deterministic)
    """
    if chart_update == None: # no chart update
        chart_update = lambda *args: args[0:2]

    # euler:
    def euler(c:[tuple[ndarray, ndarray, ndarray]],y:[tuple[ndarray, ndarray]])->tuple[ndarray, ndarray]:
        t,x,chart = c
        dt,*_ = y
        return ((t+dt,*chart_update(x+dt*ode_f(c,y[1:]),chart,y[1:])),)*2

    # Runge-kutta:
    def rk4(c:[tuple[ndarray, ndarray, ndarray]],y:[tuple[ndarray, ndarray]])->tuple[ndarray, ndarray]:
        t,x,chart = c
        dt,*_ = y
        k1 = ode_f(c,y[1:])
        k2 = ode_f((t+dt/2,x + dt/2*k1,chart),y[1:])
        k3 = ode_f((t+dt/2,x + dt/2*k2,chart),y[1:])
        k4 = ode_f((t,x + dt*k3,chart),y[1:])
        return ((t+dt,*chart_update(x + dt/6*(k1 + 2*k2 + 2*k3 + k4),chart,y[1:])),)*2

    if method == 'euler':
        return euler
    elif method == 'rk4':
        return rk4
    else:
        assert(False)

def integrate(ode:Callable[tuple[ndarray, ndarray, ndarray], ndarray],
              chart_update:bool,
              x:ndarray,chart:ndarray,dts:ndarray,*ys) -> tuple[ndarray, ndarray]:
    """return symbolic path given ode and integrator"""
    _,xs = scan(integrator(ode,chart_update),
            (0.,x,chart),
            (dts,*ys))
    return xs if chart_update is not None else xs[0:2]

def integrate_sde(sde:Callable[tuple[ndarray, ndarray, ndarray], ndarray],
                  integrator:Callable,chart_update,
                  x:ndarray,
                  chart:ndarray,
                  dts:ndarray,
                  dWs:ndarray,
                  *cy)->tuple[ndarray, ndarray]:
    """
    sde functions should return (det,sto,Sigma) where
    det is determinisitc part, sto is stochastic part,
    and Sigma stochastic generator (i.e. often sto=dot(Sigma,dW)
    """
    _,xs = scan(integrator(sde,chart_update),
            (0.,x,chart,*cy),
            (dts,dWs,))
    return xs

def integrator_stratonovich(sde_f:Callable[tuple[ndarray, ndarray, ndarray], ndarray],
                            chart_update:Callable[[ndarray, ndarray, ...], tuple[ndarray, ndarray, ...]]=None):
    """Stratonovich integration for SDE"""
    if chart_update == None: # no chart update
        chart_update = lambda xp,chart,*cy: (xp,chart,*cy)

    def euler_heun(c:[tuple[ndarray, ndarray, ndarray]],y:[tuple[ndarray, ndarray]])->tuple[ndarray, ndarray]:
        t,x,chart,*cy = c
        dt,dW = y

        (detx, stox, X, *dcy) = sde_f(c,y)
        tx = x + stox
        cy_new = tuple([y+dt*dy for (y,dy) in zip(cy,dcy)])
        return ((t+dt,*chart_update(x + dt*detx + 0.5*(stox + sde_f((t+dt,tx,chart,*cy),y)[1]), chart, *cy_new),),)*2

    return euler_heun

def integrator_ito(sde_f:Callable[tuple[ndarray, ndarray, ndarray], ndarray],
                   chart_update:Callable[[ndarray, ndarray, ...], tuple[ndarray, ndarray, ...]]=None):
    
    """Ito integration for SDE"""
    
    if chart_update == None: # no chart update
        chart_update = lambda xp,chart,*cy: (xp,chart,*cy)

    def euler(c:[tuple[ndarray, ndarray, ndarray]],y:[tuple[ndarray, ndarray]])->tuple[ndarray, ndarray]:
        t,x,chart,*cy = c
        dt,dW = y

        (detx, stox, X, *dcy) = sde_f(c,y)
        cy_new = tuple([y+dt*dy for (y,dy) in zip(cy,dcy)])
        return ((t+dt,*chart_update(x + dt*detx + stox, chart, *cy_new)),)*2

    return euler

#%% Cross Product

@jit
def cross(a:ndarray, b:ndarray)->ndarray:
    return jnp.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]])