#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 22:38:14 2023

@author: frederik
"""

#%% Sources

#%% Modules

#JAX
import jax.numpy as jnp
from jax import vmap, Array
from jax.nn import tanh

#JAXGeometry
import JAXGeometry
#from JAXGeometry.utils import *
from JAXGeometry.manifolds.Sn import *
from JAXGeometry.statistics.score_matching.ScoreTraining import train_full
from JAXGeometry.stochastics import Brownian_coords, product_sde
from JAXGeometry.stochastics.product_sde import tile

#haiku
import haiku as hk

#random
import random

#%% SN

def train_sn(file_path:str = 'models/S', 
             N:int=2, 
             loss_type:str='vsm',
             max_T:float=1.0,
             lr_rate:float=0.01,
             epochs:int=100,
             x_samples:int=2**5,
             t_samples:int=2**7,
             repeats:int=2**3,
             samples_per_batch:int=2**4,
             dt_steps:int=1000,
             save_step:int=10,
             seed:int=2712)->None:
    
    global x0s
    
    def data_generator():
        while True:
            global x0s
            _dts = dts(T=max_T, n_steps=dt_steps)
            (ts,xss,chartss,*_) = product((jnp.repeat(x0s[0],x_samples,axis=0),jnp.repeat(x0s[1],x_samples,axis=0)),
                                          _dts,dWs(N_sim*M.dim,_dts).reshape(-1,N_sim,M.dim),jnp.repeat(1.,N_sim))
            Fx0s = vmap(lambda x,chart: M.F((x,chart)))(*x0s)
            x0s = (xss[-1,::x_samples],chartss[-1,::x_samples])
           
            inds = jnp.array(random.sample(range(_dts.shape[0]), t_samples))
            ts = ts[inds]
            samples = xss[inds]
            charts = chartss[inds]
           
            yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,x_samples,axis=0),(t_samples,1)),
                             vmap(lambda x,chart: M.F((x,chart)))(samples.reshape((-1,M.dim)),charts.reshape((-1,chartss.shape[-1]))),
                             jnp.repeat(ts,N_sim).reshape((-1,1))
                            ))
    
    def update_coords(Fx:jnp.ndarray)->tuple[jnp.ndarray, jnp.ndarray]:
        
        chart = M.centered_chart(Fx)
        
        return (M.invF((Fx, chart)), chart)
    
    def tmx_fun(params:hk.Params, state_val:dict, rng_key:Array, x0:jnp.ndarray, x:jnp.ndarray, t:jnp.ndarray):
        
        return jnp.dot(M.invF((M.F(x), x[1])), model.apply(params, jnp.hstack((x0, M.F(x), t))))
    
    file_path = file_path+str(N)+'/'
    
    M = Sn(N=N, use_spherical_coords=False,chart_center=N)
    
    x0 = M.coords([0.]*N)
    
    Brownian_coords.initialize(M)
    
    x0s = tile(x0, repeats)
    (product, sde_product, chart_update_product) = product_sde.initialize(M, 
                                                                        M.sde_Brownian_coords, 
                                                                        M.chart_update_Brownian_coords)
    
    @hk.transform
    def model(x):
        
        x_new = x.T
        x1 = x_new[:M.emb_dim].T
        x2 = x_new[M.emb_dim:(2*M.emb_dim)].T
        t = x_new[-1]
        if x1.ndim==2:
            t = t.reshape(-1,1)
        model = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            lambda x: hk.Linear(M.emb_dim)(x)+(x1-x2)/t
            ])
        
        return model(x)
    
    N_sim = x_samples*repeats
    
    _dts = dts(T=max_T, n_steps=dt_steps)
    print("Hallo")
    (ts,xss,chartss,*_) = product((jnp.repeat(x0s[0],x_samples,axis=0),jnp.repeat(x0s[1],x_samples,axis=0)),
                                  _dts,dWs(N_sim*M.dim,_dts).reshape(-1,N_sim,M.dim),jnp.repeat(1.,N_sim))
    print("Hallo")
    Fx0s = vmap(lambda x,chart: M.F((x,chart)))(*x0s)
    x0s = (xss[-1,::x_samples],chartss[-1,::x_samples])
      
    inds = jnp.array(random.sample(range(_dts.shape[0]), t_samples))
    ts = ts[inds]
    samples = xss[inds]
    charts = chartss[inds]
    

    return