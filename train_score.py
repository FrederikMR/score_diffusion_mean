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

#scores
from scores import models

from ManLearn.VAE.VAE_MNIST import model as mnist_model
from ManLearn.VAE.VAE_MNIST import model_encoder as mnist_encoder
from ManLearn.VAE.VAE_MNIST import model_decoder as mnist_decoder
from ManLearn.VAE.VAE_MNIST import VAEOutput as mnist_output

from ManLearn.VAE.VAE_SVHN import model as svhn_model
from ManLearn.VAE.VAE_SVHN import model_encoder as svhn_encoder
from ManLearn.VAE.VAE_SVHN import model_decoder as svhn_decoder
from ManLearn.VAE.VAE_SVHN import VAEOutput as svhn_output

from ManLearn.VAE.VAE_CelebA import model as celeba_model
from ManLearn.VAE.VAE_CelebA import model_encoder as celeba_encoder
from ManLearn.VAE.VAE_CelebA import model_decoder as celeba_decoder
from ManLearn.VAE.VAE_CelebA import VAEOutput as celeba_output

from ManLearn.train_MNIST import load_dataset as load_mnist
from ManLearn.train_SVHN import load_dataset as load_svhn
from ManLearn.train_CelebA import load_dataset as load_celeba
from ManLearn.model_loader import load_model

#jaxgeometry
from jaxgeometry.manifolds import Euclidean, nSphere, Ellipsoid, Cylinder, S1, Torus, \
    H2, Landmarks, Heisenberg, SPDN, Latent, HypParaboloid
from jaxgeometry.setup import dts, dWs, hessianx
from jaxgeometry.statistics import score_matching
from jaxgeometry.statistics.score_matching.model_loader import load_model
from jaxgeometry.stochastics import Brownian_coords, product_sde, Brownian_sR
from jaxgeometry.stochastics.product_sde import tile

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="MNIST",
                        type=str)
    parser.add_argument('--N', default=2,
                        type=int)
    parser.add_argument('--loss_type', default="dsm",
                        type=str)
    parser.add_argument('--train_net', default="s1",
                        type=str)
    parser.add_argument('--max_T', default=1.0,
                        type=float)
    parser.add_argument('--lr_rate', default=0.001,
                        type=float)
    parser.add_argument('--epochs', default=50000,
                        type=int)
    parser.add_argument('--x_samples', default=32,
                        type=int)
    parser.add_argument('--t_samples', default=128,
                        type=int)
    parser.add_argument('--repeats', default=8,
                        type=int)
    parser.add_argument('--samples_per_batch', default=16,
                        type=int)
    parser.add_argument('--dt_steps', default=1000,
                        type=int)
    parser.add_argument('--save_step', default=10,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% Generate Data directly on the manifold

def xgenerator(M:object, 
               product:Callable[[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray],
               x_samples:int=2**5,
               t_samples:int=2**7,
               N_sim:int = 2**8,
               max_T:float=1.0, 
               dt_steps:int=1000):
    
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
        
#%% Update Coordinates, which are sampled directly on the manifold

def update_xcoords(M:object, Fx:jnp.ndarray)->tuple[jnp.ndarray, jnp.ndarray]:
    
    chart = M.centered_chart(Fx)
    
    return (Fx,chart)

#%% Apply the model and project it onto the manifold

def proj_gradx(s1_model:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray], 
               x0:jnp.ndarray, 
               x:tuple[jnp.ndarray, jnp.ndarray], 
               t:jnp.ndarray):
    
    return s1_model(x0, x[0], t)

#%% Apply the model and project it onto the manifold

def proj_hessx(s1_model:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray], 
               s2_model:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
               x0:jnp.ndarray, 
               x:tuple[jnp.ndarray, jnp.ndarray], 
               t:jnp.ndarray
               )->jnp.ndarray:
    
    return s2_model(x0,x[0],t)

#%% Generate Data in the embedded space of the manifold

def chartgenerator(M:object, 
                   product:Callable[[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   x_samples:int=2**5,
                   t_samples:int=2**7,
                   N_sim:int = 2**8,
                   max_T:float=1.0, 
                   dt_steps:int=1000):
    while True:
        global x0s
        _dts = dts(T=max_T, n_steps=dt_steps)
        dW = dWs(N_sim*M.dim,_dts).reshape(-1,N_sim,M.dim)
        (ts,xss,chartss,*_) = product((jnp.repeat(x0s[0],x_samples,axis=0),jnp.repeat(x0s[1],x_samples,axis=0)),
                                      _dts,dW,jnp.repeat(1.,N_sim))
        Fx0s = vmap(lambda x,chart: M.F((x,chart)))(*x0s)
        x0s = (xss[-1,::x_samples],chartss[-1,::x_samples])
       
        inds = jnp.array(random.sample(range(_dts.shape[0]), t_samples))
        ts = ts[inds]
        samples = xss[inds]
        charts = chartss[inds]
       
        yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,x_samples,axis=0),(t_samples,1)),
                         vmap(lambda x,chart: M.F((x,chart)))(samples.reshape((-1,M.dim)),charts.reshape((-1,chartss.shape[-1]))),
                         jnp.repeat(ts,N_sim).reshape((-1,1)),
                         dW[inds].reshape(-1,M.dim),
                         jnp.repeat(_dts[inds],N_sim).reshape((-1,1)),
                        ))
        
#%% Update Coordinates, which are sampled in the embedded space

def update_chartcoords(M:object, Fx:jnp.ndarray)->tuple[jnp.ndarray, jnp.ndarray]:
    
    chart = M.centered_chart(Fx)
    
    return (M.invF((Fx,chart)),chart)

#%% Apply the tangent in embedded space onto the manifold

def proj_gradchart(M:object,
                   s1_model:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   x0:jnp.ndarray, 
                   x:tuple[jnp.ndarray, jnp.ndarray], 
                   t:jnp.ndarray):
    
    Fx = M.F(x)

    return jnp.dot(M.invJF((Fx,x[1])), s1_model(x0,Fx,t))

#%% Apply the model and project it onto the manifold

def proj_hesschart(M:object,
                   s1_model:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray], 
                   s2_model:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   x0:jnp.ndarray, 
                   x:tuple[jnp.ndarray, jnp.ndarray], 
                   t:jnp.ndarray
                   )->jnp.ndarray:
    
    Hf = hessianx(M.invF)
    Fx = M.F(x)

    term1 = jnp.dot(M.invJF((Fx,x[1])), s2_model(x0,Fx,t).reshape(M.emb_dim, M.emb_dim))
    term2 = jnp.dot(Hf((Fx,x[1])), s1_model(x0,Fx,t))
    
    return term1+term2 

#%% train for (x,y,t)

def trainxt(manifold:str="RN",
            N:int=2, 
            loss_type:str='dsm',
            train_net:str="s1",
            max_T:float=1.0,
            lr_rate:float=0.001,
            epochs:int=50000,
            x_samples:int=2**5,
            t_samples:int=2**7,
            repeats:int=2**3,
            samples_per_batch:int=2**4,
            dt_steps:int=1000,
            save_step:int=10,
            seed:int=2712
            )->None:
    
    global x0s
    
    N_sim = x_samples*repeats
    
    if manifold == "RN":
        s1_path = ''.join(('scores/R',str(N),'/',loss_type,'/'))
        s2_path = ''.join(('scores/R',str(N),'/s2/'))
        M = Euclidean(N=N)
        Brownian_coords(M)
        
        N_dim = M.dim
        x0 = M.coords([0.]*N)
        
        if N<10:
            layers = [50,100,100,50]
        elif N<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : xgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_xcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradx(s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hessx(s1, s2, x0, x, t)
    
    elif manifold == "S1":
        s1_path = ''.join(('scores/S1/',loss_type,'/'))
        s2_path = ''.join(('scores/S1/s2/'))
        
        M = S1(use_spherical_coords=True)
        Brownian_coords(M)
        
        N_dim = M.emb_dim
        x0 = M.coords([0.])
        layers = [50,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : chartgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_chartcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradchart(M, s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hesschart(M, s1, s2, x0, x, t)
        
    elif manifold == "SN":
        
        s1_path = ''.join(('scores/S',str(N),'/',loss_type,'/'))
        s2_path = ''.join(('scores/S',str(N),'/s2'))
        M = nSphere(N=N)
        Brownian_coords(M)
        
        N_dim = M.emb_dim
        x0 = M.coords([0.]*N)
        
        if N<10:
            layers = [50,100,100,50]
        elif N<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : chartgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_chartcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradchart(M, s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hesschart(M, s1, s2, x0, x, t)
        
    elif manifold == "Ellipsoid":
        
        s1_path = ''.join(('scores/Ellipsoid',str(N),'/',loss_type,'/'))
        s2_path = ''.join(('scores/Ellipsoid',str(N),'/s2/'))
        M = Ellipsoid(N=N, params = jnp.linspace(0.5,1.0,N+1))
        Brownian_coords(M)
        
        N_dim = M.emb_dim
        x0 = M.coords([0.]*N)
        
        if N<10:
            layers = [50,100,100,50]
        elif N<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : chartgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_chartcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradchart(M, s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hesschart(M, s1, s2, x0, x, t)
        
    elif manifold == "Cylinder":
        
        s1_path = ''.join(('scores/Cylinder/', loss_type,'/'))
        s2_path = 'scores/Cylinder/s2/'
        M = Cylinder(params=(1.,jnp.array([0.,0.,1.]),jnp.pi/2.))
        Brownian_coords(M)
        
        N_dim = M.emb_dim
        x0 = M.coords([0.]*2)
        layers = [50,100,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : chartgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_chartcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradchart(M, s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hesschart(M, s1, s2, x0, x, t)
        
    elif manifold == "Torus":

        s1_path = ''.join(('scores/Torus/', loss_type,'/'))
        s2_path = 'scores/Torus/s2/'
        M = Torus()
        Brownian_coords(M)
        
        N_dim = M.emb_dim
        x0 = M.coords([0.]*2)
        layers = [50,100,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : chartgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_chartcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradchart(M, s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hesschart(M, s1, s2, x0, x, t)
        
    elif manifold == "Landmarks":
        
        s1_path = ''.join(('scores/Landmarks', str(N), '/',loss_type,'/'))
        s2_path = ''.join(('scores/Landmarks', str(N), '/s2/'))
        M = Landmarks(N=N,m=2)
        Brownian_coords(M)
        
        N_dim = M.dim
        x0 = M.coords(jnp.vstack((jnp.linspace(-5.0,5.0,M.N),jnp.zeros(M.N))).T.flatten())
        
        if 2*N<10:
            layers = [50,100,100,50]
        elif 2*N<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : xgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_xcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradx(s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hessx(s1, s2, x0, x, t)
        
    elif manifold == "SPDN":

        s1_path = ''.join(('scores/SPDN', str(N), '/',loss_type,'/'))
        s2_path = ''.join(('scores/SPDN', str(N), '/s2/'))
        M = SPDN(N=N)
        
        Brownian_coords(M)
        
        N_dim = M.emb_dim
        x0 = M.coords([0.]*(N*(N+1)//2))
        if N*N<10:
            layers = [50,100,100,50]
        elif N*N<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : chartgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_chartcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradchart(M, s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hesschart(M, s1, s2, x0, x, t)
        
    elif manifold == "HypParaboloid":

        s1_path = ''.join(('scores/HypParaboloid/',loss_type,'/'))
        s2_path = 'scores/HypParaboloid/s2/'
        M = HypParaboloid()
        Brownian_coords(M)
        
        N_dim = M.dim
        x0 = M.coords([0.]*N)
        
        if N<10:
            layers = [50,100,100,50]
        elif N<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : xgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_xcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradx(s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hessx(s1, s2, x0, x, t)
        
    elif manifold == "MNIST":
        
        s1_path = ''.join(('scores/MNIST/',loss_type,'/'))
        s2_path = 'scores/MNIST/s2/'
        
        ds = load_mnist("train", 100, 2712)
        
        state = load_model('ManLearn/models/MNIST/VAE/')
        F = lambda x: mnist_decoder.apply(state.params, state.rng_key, x[0]).reshape(-1)
        
        M = Latent(F=F,dim=2,emb_dim=28*28,invF=None)
        Brownian_coords(M)
        
        N_dim = M.dim
        x0 = mnist_encoder.apply(state.params, state.rng_key, next(ds).image)
        x0 = M.coords(x0[0])
        
        layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : xgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_xcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradx(s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hessx(s1, s2, x0, x, t)
        
    elif manifold == "SVHN":
        
        s1_path = ''.join(('scores/SVHN/',loss_type,'/'))
        s2_path = 'scores/SVHN/s2/'
        
        ds = load_svhn()
        
        state = load_model('ManLearn/models/SVHN/VAE/')
        F = lambda x: svhn_decoder.apply(state.params, state.rng_key, x[0]).reshape(-1)
        
        M = Latent(F=F,dim=32,emb_dim=32*32*3,invF=None)
        Brownian_coords(M)
        
        N_dim = M.dim
        x0 = svhn_encoder.apply(state.params, state.rng_key, next(ds).image)
        x0 = M.coords(x0[0])
        
        layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : xgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_xcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradx(s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hessx(s1, s2, x0, x, t)
        
    elif manifold == "CelebA":
        
        s1_path = ''.join(('scores/CelebS/',loss_type,'/'))
        s2_path = 'scores/CelebS/s2/'
        
        ds = load_celeba()
        
        state = load_model('ManLearn/models/CelebA/VAE/')
        F = lambda x: celeba_decoder.apply(state.params, state.rng_key, x[0]).reshape(-1)
        
        M = Latent(F=F,dim=32,emb_dim=64*64*3,invF=None)
        Brownian_coords(M)
        
        N_dim = M.dim
        x0 = svhn_encoder.apply(state.params, state.rng_key, next(ds).image)
        x0 = M.coords(x0[0])
        
        layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=N_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=N_dim, r = max(N_dim//2,1))(x))
        
        data_generator = lambda : xgenerator(M, 
                                            product,
                                            x_samples=x_samples,
                                            t_samples=t_samples,
                                            N_sim = N_sim,
                                            max_T = max_T, 
                                            dt_steps = dt_steps)
        update_coords = lambda Fx: update_xcoords(M, Fx)
        proj_grad = lambda s1, x0, x, t: proj_gradx(s1, x0, x,t)
        proj_hess = lambda s1, s2, x0, x, t: proj_hessx(s1, s2, x0, x, t)
        
    else:
        return
        
        
    x0s = tile(x0, repeats)
    (product, sde_product, chart_update_product) = product_sde(M, 
                                                               M.sde_Brownian_coords, 
                                                               M.chart_update_Brownian_coords)
    
    if train_net == "s2":
        state = load_model(s1_path)
        rng_key = jran.PRNGKey(2712)
        s1 = lambda x,y,t: s1_model.apply(state.params,rng_key, jnp.hstack((x, y, t)))
        
        score_matching.trainxt.train_s2(M=M,
                                        s1_model = s1,
                                        s2_model = s2_model,
                                        data_generator=data_generator,
                                        update_coords=update_coords,
                                        proj_grad=proj_grad,
                                        proj_hess=proj_hess,
                                        N_dim=N_dim,
                                        batch_size=x_samples*t_samples*repeats,
                                        lr_rate = lr_rate,
                                        epochs=epochs,
                                        save_step=save_step,
                                        save_path = s2_path,
                                        seed=seed
                                        )
    else:
        score_matching.trainxt.train_s1(M=M,
                                        model=s1_model,
                                        data_generator=data_generator,
                                        update_coords=update_coords,
                                        proj_grad=proj_grad,
                                        N_dim=N_dim,
                                        batch_size=x_samples*t_samples*repeats,
                                        epochs=epochs,
                                        save_step=save_step,
                                        save_path = s1_path,
                                        loss_type=loss_type,
                                        seed=seed
                                        )
    
    
    return

#%% Main

if __name__ == '__main__':
    
    args = parse_args()
    
    trainxt(manifold=args.manifold,
            N=args.N, 
            loss_type=args.loss_type,
            train_net=args.train_net,
            max_T=args.max_T,
            lr_rate=args.lr_rate,
            epochs=args.epochs,
            x_samples=args.x_samples,
            t_samples=args.t_samples,
            repeats=args.repeats,
            samples_per_batch=args.samples_per_batch,
            dt_steps=args.dt_steps,
            save_step=args.save_step,
            seed=args.seed)
    