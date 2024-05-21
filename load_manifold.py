#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:09:44 2024

@author: fmry
"""

#%% Sources

#%% Modules

#jax
import jax.numpy as jnp

from gp.gp import RM_EG

from typing import List, Tuple

#jaxgeometry
from jaxgeometry.manifolds import *

#%% Get generator by dimension

def get_generator_dim(manifold:str, dim:int)->Tuple[List,List]:
    
    if manifold == "Euclidean":
        layers_s1 = [128, 128, 128]
        layers_s2 = [128,128,128]
        #if dim < 15:
        #    layers_s1 = [128, 128, 128]
        #    layers_s2 = [32,32,32]
        #else:
        #    layers_s1 = [512, 512, 512, 512, 512]
        #    layers_s2 = [128, 128, 128, 128, 128]
    elif manifold == "Sphere":
        layers_s1 = [512, 512, 512, 512, 512]
        layers_s2 = [512, 512, 512, 512, 512]
    else:
        layers_s1 = [512, 512, 512]
        layers_s2 = [512, 512, 512]
        
    return layers_s1, layers_s2

#%% Load Manifold

def load_manifold(manifold:str, dim:int=None)->None:

    if manifold == "Euclidean":
        sampling_method = 'LocalSampling'
        M = Euclidean(N=dim)
        generator_dim = M.dim
        x0 = M.coords([0.]*dim)
        opt_val = "opt"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Circle":
        sampling_method = 'TMSampling'
        M = S1()
        generator_dim = M.emb_dim
        x0 = M.coords([0.])
        opt_val = "gradient"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Sphere":
        sampling_method = 'TMSampling'
        M = nSphere(N=dim)
        generator_dim = M.emb_dim
        x0 = M.coords([0.]*dim)
        opt_val = "gradient"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "H2":
        sampling_method = 'LocalSampling'
        M = H2()
        generator_dim = M.dim
        x0 = M.coords([0.]*M.dim)
        layers = get_generator_dim(manifold, generator_dim)
        opt_val = "x0"
    elif manifold == "HyperbolicSpace":
        sampling_method = 'TMSampling'
        M = nHyperbolicSpace(N=dim)
        generator_dim = M.emb_dim
        x0 = (jnp.concatenate((jnp.zeros(dim-1), -1.*jnp.ones(1))),)*2
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Grassmanian":
        sampling_method = 'TMSampling'
        M = Grassmanian(N=2*dim,K=dim)
        generator_dim = M.emb_dim
        x0 = (jnp.eye(2*dim)[:,:dim].reshape(-1),)*2
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "SO":
        sampling_method = 'TMSampling'
        M = SO(N=dim)
        generator_dim = M.emb_dim
        x0 = (jnp.eye(dim).reshape(-1),)*2
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Stiefel":
        sampling_method = 'TMSampling'
        M = Stiefel(N=dim, K=2)
        generator_dim = M.emb_dim
        x0 = (jnp.block([jnp.eye(2), jnp.zeros((2,dim-2))]).T.reshape(-1),)*2
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Ellipsoid":
        sampling_method = 'EmbeddedSampling'
        M = nEllipsoid(N=dim, params = jnp.linspace(0.5,1.0,dim+1))
        generator_dim = M.emb_dim
        x0 = M.coords([0.]*dim)
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Cylinder":
        sampling_method = 'EmbeddedSampling'
        M = Cylinder(params=(1.,jnp.array([0.,0.,1.]),jnp.pi/2.))
        generator_dim = M.emb_dim
        x0 = M.coords([0.]*2)
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Torus":
        sampling_method = 'EmbeddedSampling'
        M = Torus()        
        generator_dim = M.emb_dim
        x0 = M.coords([0.]*2)
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Landmarks":
        sampling_method = 'LocalSampling'
        M = Landmarks(N=dim,m=2)   
        generator_dim = M.dim
        x0 = M.coords(jnp.vstack((jnp.linspace(-5.0,5.0,M.N),jnp.linspace(0.0,0.0,M.N))).T.flatten())
        if dim >=10:
            with open('../../../Data/landmarks/Papilonidae/Papilionidae_landmarks.txt', 'r') as the_file:
                all_data = [line.strip() for line in the_file.readlines()]
                
                x1 = jnp.array([float(x) for x in all_data[0].split()[2:]])
                x2 = jnp.array([float(x) for x in all_data[1].split()[2:]])
                
                idx = jnp.round(jnp.linspace(0, len(x1) - 1, dim)).astype(int)
                x0 = M.coords(jnp.vstack((x1[idx],x2[idx])).T.flatten())
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "SPDN":
        sampling_method = 'LocalSampling'
        M = SPDN(N=dim)
        generator_dim = M.dim
        #x0 = M.coords([1.]*(dim*(dim+1)//2))
        #x0 = (x0[0], M.F(x0))
        x0 = 10.0*jnp.eye(dim)
        x0 = (M.invF((x0,x0)), x0.reshape(-1))
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "Sym":
        sampling_method = 'LocalSampling'
        M = Sym(N=dim)
        generator_dim = M.dim
        x0 = M.coords([1.]*(dim*(dim+1)//2))
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == "HypParaboloid":
        sampling_method = 'LocalSampling'
        M = HypParaboloid()
        generator_dim = M.dim
        x0 = M.coords([0.]*2)
        opt_val = "x0"
        layers = get_generator_dim(manifold, generator_dim)
    elif manifold == 'gp_mnist':
        
        default_omega = 500.
        
        def k_fun(x,y, beta=1.0, omega=default_omega):
    
            x_diff = x-y
            
            return beta*jnp.exp(-omega*jnp.dot(x_diff, x_diff)/2)

        def Dk_fun(x,y, beta=1.0, omega=default_omega):
            
            x_diff = y-x
            
            return omega*x_diff*k_fun(x,y,beta,omega)
        
        def DDk_fun(x,y, beta=1.0, omega=default_omega):
            
            N = len(x)
            x_diff = (x-y).reshape(1,-1)
            
            return -omega*k_fun(x,y,beta,omega)*(x_diff.T.dot(x_diff)*omega-jnp.eye(N))
        
        rot = jnp.load('Data/MNIST/rot.npy')
        num_rotate = len(rot)

        theta = jnp.linspace(0,2*jnp.pi,num_rotate)
        x1 = jnp.cos(theta)
        x2 = jnp.sin(theta)
        
        sigman = 0.0
        X_training = jnp.vstack((x1,x2))
        y_training = rot.reshape(num_rotate, -1).T
        RMEG = RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, 
                     Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)

        g = lambda x: RMEG.G(x[0])
        
        M = LearnedManifold(g,N=2)
        generator_dim = M.dim
        x0 = M.coords(jnp.array([jnp.cos(0.), jnp.sin(0.)]))
        sampling_method = 'LocalSampling'
        layers = get_generator_dim(manifold, generator_dim)
        opt_val = 'x0'
    else:
        return

    return M, x0, sampling_method, generator_dim, layers, opt_val