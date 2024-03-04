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

import os

#argparse
import argparse

#jaxgeometry
from jaxgeometry.manifolds import *
from jaxgeometry.statistics.score_matching import TMSampling, LocalSampling, \
    EmbeddedSampling, ProjectionSampling

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="SPDN",
                        type=str)
    parser.add_argument('--dim', default=5,
                        type=int)
    parser.add_argument('--N_sim', default=100,
                        type=int)
    parser.add_argument('--sampling_method', default='LocalSampling',
                        type=str)
    parser.add_argument('--save_path', default='../data/',
                        type=str)
    parser.add_argument('--max_T', default=0.5,
                        type=float)
    parser.add_argument('--dt_steps', default=1000,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% train for (x,y,t)

def train_score()->None:
    
    args = parse_args()
    
    save_path = f"{args.save_path}{args.manifold}{args.dim}/"
    
    if args.manifold == "Euclidean":
        sampling_method = 'LocalSampling'
        M = Euclidean(N=args.dim)
        x0 = M.coords([0.]*args.dim)
    elif args.manifold == "Circle":
        sampling_method = 'TMSampling'
        M = S1()
        x0 = M.coords([0.])
    elif args.manifold == "Sphere":
        sampling_method = 'TMSampling'
        M = nSphere(N=args.dim)
        x0 = M.coords([0.]*args.dim)
    elif args.manifold == "HyperbolicSpace":
        sampling_method = 'TMSampling'
        M = nHyperbolicSpace(N=args.dim)
        x0 = (jnp.concatenate((jnp.zeros(args.dim-1), -1.*jnp.ones(1))),)*2
    elif args.manifold == "Grassmanian":
        sampling_method = 'TMSampling'
        M = Grassmanian(N=2*args.dim,K=args.dim)
        x0 = (jnp.eye(2*args.dim)[:,:args.dim].reshape(-1),)*2
    elif args.manifold == "SO":
        sampling_method = 'TMSampling'
        M = SO(N=args.dim)
        x0 = (jnp.eye(args.dim).reshape(-1),)*2
    elif args.manifold == "Stiefel":
        sampling_method = 'TMSampling'
        M = Stiefel(N=args.dim, K=2)
        x0 = (jnp.block([jnp.eye(2), jnp.zeros((2,args.dim-2))]).T.reshape(-1),)*2
    elif args.manifold == "Ellipsoid":
        sampling_method = 'TMSampling'
        M = nEllipsoid(N=args.dim, params = jnp.linspace(0.5,1.0,args.dim+1))
        x0 = M.coords([0.]*args.dim)
    elif args.manifold == "Cylinder":
        sampling_method = 'EmbeddedSampling'
        M = Cylinder(params=(1.,jnp.array([0.,0.,1.]),jnp.pi/2.))
        x0 = M.coords([0.]*2)
    elif args.manifold == "Torus":
        sampling_method = 'EmbeddedSampling'
        M = Torus()        
        x0 = M.coords([0.]*2)
    elif args.manifold == "Landmarks":
        sampling_method = 'LocalSampling'
        M = Landmarks(N=args.dim,m=2)        
        x0 = M.coords(jnp.vstack((jnp.linspace(-10.0,10.0,M.N),jnp.linspace(10.0,-10.0,M.N))).T.flatten())
        if args.dim >=10:
            with open('../../Data/landmarks/Papilonidae/Papilionidae_landmarks.txt', 'r') as the_file:
                all_data = [line.strip() for line in the_file.readlines()]
                
                x1 = jnp.array([float(x) for x in all_data[0].split()[2:]])
                x2 = jnp.array([float(x) for x in all_data[1].split()[2:]])
                
                idx = jnp.round(jnp.linspace(0, len(x1) - 1, args.dim)).astype(int)
                x0 = M.coords(jnp.vstack((x1[idx],x2[idx])).T.flatten())
    elif args.manifold == "SPDN":
        sampling_method = 'LocalSampling'
        M = SPDN(N=args.dim)
        x0 = M.coords([10.]*(args.dim*(args.dim+1)//2))
    elif args.manifold == "Sym":
        sampling_method = 'LocalSampling'
        M = Sym(N=args.dim)
        x0 = M.coords([1.]*(args.dim*(args.dim+1)//2))
    elif args.manifold == "HypParaboloid":
        sampling_method = 'LocalSampling'
        M = HypParaboloid()
        x0 = M.coords([0.]*2)
    else:
        return
        
    if args.sampling_method == 'LocalSampling':
        generator_dim = M.dim
        data_generator = LocalSampling(M=M,
                                       x0=x0,
                                       max_T=args.max_T,
                                       dt_steps=args.dt_steps,
                                       )
        sim = data_generator.sim_diffusion_mean((x0[0],x0[1]), args.N_sim)
    elif args.sampling_method == "EmbeddedSampling":
        generator_dim = M.dim
        data_generator = EmbeddedSampling(M=M,
                                          x0=x0,
                                          max_T=args.max_T,
                                          dt_steps=args.dt_steps,
                                          )
        sim = data_generator.sim_diffusion_mean((x0[0],x0[1]), args.N_sim)
    elif args.sampling_method == "ProjectionSampling":
        generator_dim = M.emb_dim
        data_generator = ProjectionSampling(M=M,
                                            x0=(x0[1],x0[0]),
                                            dim=generator_dim,
                                            max_T=args.max_T,
                                            dt_steps=args.dt_steps,
                                            )
        sim = data_generator.sim_diffusion_mean((x0[1],x0[0]), args.N_sim)
    elif args.sampling_method == "TMSampling":
        generator_dim = M.emb_dim
        data_generator = TMSampling(M=M,
                                    x0=(x0[1],x0[0]),
                                    dim=generator_dim,
                                    Exp_map=lambda x, v: M.ExpEmbedded(x[0],v),
                                    max_T=args.max_T,
                                    dt_steps=args.dt_steps,
                                    )
        sim = data_generator.sim_diffusion_mean((x0[1],x0[0]), args.N_sim)
        
        
    xs, chart = sim[0], sim[1]
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    np.savetxt(''.join((save_path, 'xs.csv')), xs, delimiter=",")
    np.savetxt(''.join((save_path, 'chart.csv')), chart, delimiter=",")
    
    return

#%% Main

if __name__ == '__main__':
        
    train_score()
    
