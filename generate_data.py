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

import numpy as np

from scipy import ndimage

from gp.gp import RM_EG

import os

#argparse
import argparse

from load_manifold import load_manifold

#jaxgeometry
from jaxgeometry.manifolds import *
from jaxgeometry.statistics.score_matching import RiemannianBrownianGenerator
from ManLearn.train_MNIST import load_dataset as load_mnist

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="HypParaboloid",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--T', default=0.5,
                        type=float)
    parser.add_argument('--dt_steps', default=1000,
                        type=int)
    parser.add_argument('--N_sim', default=1000,
                        type=int)
    parser.add_argument('--save_path', default='../data/',
                        type=str)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% train for (x,y,t)

def train_score()->None:
    
    args = parse_args()
    
    save_path = f"{args.save_path}{args.manifold}{args.dim}/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if args.manifold == 'gp_mnist':
        
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
        
        data_generator = load_mnist('train[:80%]', 100, 2712)
        img_base = next(data_generator).image[0]
        
        num_rotate = 200
        theta = jnp.linspace(0,2*jnp.pi,num_rotate)
        x1 = jnp.cos(theta)
        x2 = jnp.sin(theta)
        
        theta_degrees = theta*180/jnp.pi
        
        rot = []
        for v in theta_degrees:
            rot.append(ndimage.rotate(img_base, v, reshape=False))
        rot = jnp.stack(rot)/255
        
        jnp.save('Data/MNIST/rot.npy', rot)
        return
    else:
        M, x0, sampling_method, generator_dim, layers, opt_val = load_manifold(args.manifold,
                                                                               args.dim)
        
        
    data_generator = RiemannianBrownianGenerator(M=M,
                                                 x0 = x0,
                                                 dim = generator_dim,
                                                 Exp_map = lambda x, v: M.ExpEmbedded(x[0],v),
                                                 method = sampling_method,
                                                 T = args.T,
                                                 seed = args.seed
                                                 )
    sim = data_generator.sim_diffusion_mean(x0, args.N_sim)

    xs, chart = sim[0], sim[1]
    
    np.savetxt(''.join((save_path, 'xs.csv')), xs, delimiter=",")
    np.savetxt(''.join((save_path, 'chart.csv')), chart, delimiter=",")
    
    return

#%% Main

if __name__ == '__main__':
        
    train_score()
    
