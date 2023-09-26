#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:11:07 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax
import jax.numpy as jnp
from jax import vmap, Array
import jax.random as jran
from jax.nn import tanh

#haiku
import haiku as hk

#random
import random

#typing
from typing import Callable

#dataclasses
import dataclasses

#argparse
import argparse

#jaxgeometry
from jaxgeometry.manifolds import Euclidean
from jaxgeometry.setup import dts, dWs
from jaxgeometry.statistics.score_matching.trainxt import train_s1, train_s2
from jaxgeometry.statistics.score_matching.model_loader import load_model
from jaxgeometry.stochastics import Brownian_coords, product_sde
from jaxgeometry.stochastics.product_sde import tile

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--model', default="RN",
                        type=str)
    parser.add_argument('--order', default="s2",
                        type=str)
    parser.add_argument('--dim', default=2, #'trained_models/surface_R2'
                        type=int)

    args = parser.parse_args()
    return args

#%% Models

@dataclasses.dataclass
class RN_s1(hk.Module):
    
    dim:int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        x_new = x.T
        x1 = x_new[:self.dim].T
        x2 = x_new[self.dim:(2*self.dim)].T
        t = x_new[-1]
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x_new[-1].reshape(shape)
            
        grad_euc = (x1-x2)/t
        model = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            #hk.Linear(200), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            lambda x: hk.Linear(self.dim)(x)
            ])
      
        return model(x)+grad_euc
    
@dataclasses.dataclass
class RN_s2(hk.Module):
    
    dim:int = 2
    r:int = max(dim // 2,1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        model_alpha = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            #hk.Linear(200), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            hk.Linear(self.dim)
            ])
        
        model_beta = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            #hk.Linear(200), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            lambda x: hk.Linear(self.dim*self.r)(x).reshape(-1,self.dim, self.r)
            ])
        
        beta = model_beta(x)
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x.T[-1].reshape(shape)

        hess_rn = -jnp.einsum('ij,...i->...ij', jnp.eye(self.dim), 1/t)
        
        return jnp.diag(model_alpha(x))+jnp.einsum('...ik,...jk->...ij', beta, beta)+\
            hess_rn

    
@dataclasses.dataclass
class RN_model(hk.Module):
    
    dim:int = 2
    r:int = max(dim // 2,1)
    
    def s1(self, x:jnp.ndarray) -> jnp.ndarray:
        
        x_new = x.T
        x1 = x_new[:self.dim].T
        x2 = x_new[self.dim:(2*self.dim)].T
        t = x_new[-1]
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x_new[-1].reshape(shape)
            
        grad_euc = (x1-x2)/t
        model = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            #hk.Linear(200), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            lambda x: hk.Linear(self.dim)(x)
            ])
      
        return model(x)+grad_euc
    
    def s2(self, x:jnp.ndarray) -> jnp.ndarray:
        
        model_alpha = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            #hk.Linear(200), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            hk.Linear(self.dim)
            ])
        
        model_beta = hk.Sequential([
            hk.Linear(50), tanh,
            hk.Linear(100), tanh,
            #hk.Linear(200), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(400), tanh,
            #hk.Linear(200), tanh,
            hk.Linear(100), tanh,
            hk.Linear(50), tanh,
            lambda x: hk.Linear(self.dim*self.r)(x).reshape(-1,self.dim, self.r)
            ])
        
        beta = model_beta(x)
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x.T[-1].reshape(shape)

        hess_rn = -jnp.einsum('ij,...i->...ij', jnp.eye(self.dim), 1/t)
        
        return jnp.diag(model_alpha(x))+jnp.einsum('...ik,...jk->...ij', beta, beta)+\
            hess_rn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        return self.s1(x), self.s2(x)

#%% Euclidean

def train_rn_s1(file_path:str = 'models/R', 
                N:int=2, 
                loss_type:str='vsm',
                max_T:float=1.0,
                lr_rate:float=0.01,
                epochs:int=50000,
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
            dW = dWs(N_sim*M.dim,_dts).reshape(-1,N_sim,M.dim)
            (ts,xss,chartss,*_) = product((jnp.repeat(x0s[0],x_samples,axis=0),jnp.repeat(x0s[1],x_samples,axis=0)),
                                          _dts,dW,jnp.repeat(1.,N_sim))
            Fx0s = x0s[0]
            x0s = (xss[-1,::x_samples],chartss[-1,::x_samples])
           
            inds = jnp.array(random.sample(range(_dts.shape[0]), t_samples))
            ts = ts[inds]
            samples = xss[inds]
           
            yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,x_samples,axis=0),(t_samples,1)),
                             samples.reshape(-1,M.dim),
                             jnp.repeat(ts,N_sim).reshape((-1,1)),
                             dW[inds].reshape(-1,M.dim),
                             jnp.repeat(_dts[inds],N_sim).reshape((-1,1)),
                            ))
    
    def update_coords(Fx:jnp.ndarray)->tuple[jnp.ndarray, jnp.ndarray]:
        
        chart = M.centered_chart(Fx)
        
        return (Fx,chart)
    
    def tmx_fun(params:hk.Params, state_val:dict, rng_key:Array, x0:jnp.ndarray, 
                x:tuple[jnp.ndarray, jnp.ndarray], 
                t:jnp.ndarray):
        
        return model.apply(params, rng_key, jnp.hstack((x0, x[0], t)))
    
    @hk.transform
    def model(x):
        
        score = RN_s1(M.dim)
        
        return score(x)
    
    file_path = ''.join((file_path,str(N),'/',loss_type,'/'))
    N_sim = x_samples*repeats
    
    M = Euclidean(N=N)
    
    x0 = M.coords([0.]*N)
    
    Brownian_coords(M)
    
    x0s = tile(x0, repeats)
    (product, sde_product, chart_update_product) = product_sde(M, 
                                                               M.sde_Brownian_coords, 
                                                               M.chart_update_Brownian_coords)
    
    train_s1(M=M,
             model=model,
             data_generator=data_generator,
             update_coords=update_coords,
             tmx_fun=tmx_fun,
             N_dim=N,
             batch_size=x_samples*t_samples*repeats,
             epochs=epochs,
             save_step=save_step,
             optimizer=None,
             save_path = file_path,
             loss_type=loss_type,
             seed=2712
             )

    return

def train_rn_s2(file_path:str = 'models/R', 
                N:int=2, 
                gamma:float=1.0,
                max_T:float=1.0,
                lr_rate:float=0.0002,
                epochs:int=50000,
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
            dW = dWs(N_sim*M.dim,_dts).reshape(-1,N_sim,M.dim)
            (ts,xss,chartss,*_) = product((jnp.repeat(x0s[0],x_samples,axis=0),jnp.repeat(x0s[1],x_samples,axis=0)),
                                          _dts,dW,jnp.repeat(1.,N_sim))
            Fx0s = x0s[0]
            x0s = (xss[-1,::x_samples],chartss[-1,::x_samples])
           
            inds = jnp.array(random.sample(range(_dts.shape[0]), t_samples))
            ts = ts[inds]
            samples = xss[inds]
           
            yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,x_samples,axis=0),(t_samples,1)),
                             samples.reshape(-1,M.dim),
                             jnp.repeat(ts,N_sim).reshape((-1,1)),
                             dW[inds].reshape(-1,M.dim),
                             jnp.repeat(_dts[inds],N_sim).reshape((-1,1)),
                            ))
    
    @hk.transform
    def s1_model(x):
        
        score = RN_s1(M.dim)
        
        return score(x)
    
    @hk.transform
    def s2_model(x):
        
        score = RN_s2(M.dim)
        
        return score(x)
    
    file_path = ''.join((file_path,str(N),'/s2/'))
    N_sim = x_samples*repeats
    
    M = Euclidean(N=N)
    
    x0 = M.coords([0.]*N)
    
    Brownian_coords(M)
    
    x0s = tile(x0, repeats)
    (product, sde_product, chart_update_product) = product_sde(M, 
                                                               M.sde_Brownian_coords, 
                                                               M.chart_update_Brownian_coords)

    state = load_model(''.join(('models/R',str(N),'/dsm/')))
    rng_key = jran.PRNGKey(2712)
    s1 = lambda x,y,t: s1_model.apply(state.params,rng_key, jnp.hstack((x, y, t)))
    
    train_s2(M=M,
             s1_model = s1,
             s2_model = s2_model,
             data_generator=data_generator,
             N_dim=N,
             batch_size=x_samples*t_samples*repeats,
             gamma=gamma,
             lr_rate = lr_rate,
             epochs=epochs,
             save_step=save_step,
             optimizer=None,
             save_path = file_path,
             seed=2712
             )

    return


#%% m-Sphere

#%% Main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.model == "RN":
        if args.order == "s2":
            train_rn_s2(file_path = 'models/R', 
                            N=args.dim, 
                            gamma=1.0,
                            max_T=1.0,
                            lr_rate=0.0002,
                            epochs=50000,
                            x_samples=2**5,
                            t_samples=2**7,
                            repeats=2**3,
                            samples_per_batch=2**4,
                            dt_steps=1000,
                            save_step=10,
                            seed=2712)
        elif args.order == "s1":
            train_rn_s1(file_path='models/R', 
                        N=args.dim, 
                        loss_type='dsm',
                        max_T=1.0,
                        lr_rate=0.0002,
                        epochs=50000,
                        x_samples=2**5,
                        t_samples=2**7,
                        repeats=2**3,
                        samples_per_batch=2**4,
                        dt_steps=1000,
                        save_step=10,
                        seed=2712)












