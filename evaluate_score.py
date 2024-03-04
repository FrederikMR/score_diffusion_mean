#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:30:00 2024

@author: fmry
"""

#%% Sources

#%% Modules

#jax
import jax.numpy as jnp
import jax.random as jran

#argparse
import argparse

#scores
from models import models

#haiku
import haiku as hk

#pandas
import pandas as pd

#os
import os

from typing import List

#jaxgeometry
from jaxgeometry.manifolds import *
from jaxgeometry.autodiff import jacfwdx
from jaxgeometry.statistics.score_matching import train_s1, train_s2, train_s1s2, TMSampling, LocalSampling, \
    EmbeddedSampling, ProjectionSampling
from jaxgeometry.stochastics import Brownian_coords
from jaxgeometry.statistics.score_matching.model_loader import load_model
from jaxgeometry.statistics.score_matching import diffusion_mean as dm_score
from jaxgeometry.statistics.score_matching import ScoreEvaluation
from jaxgeometry.statistics import diffusion_mean as dm_bridge

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="SPDN",
                        type=str)
    parser.add_argument('--dim', default=[2,5],
                        type=List)
    parser.add_argument('--loss_type', default="dsmvr",
                        type=str)
    parser.add_argument('--s2_approx', default=0,
                        type=int)
    parser.add_argument('--fixed_time', default=0,
                        type=int)
    parser.add_argument('--s2_type', default="s1s2",
                        type=str)
    parser.add_argument('--data_path', default='../data/',
                        type=str)
    parser.add_argument('--save_path', default='../data/',
                        type=str)
    parser.add_argument('--t', default=0.1,
                        type=float)
    parser.add_argument('--t0', default=0.2,
                        type=float)
    parser.add_argument('--step_size', default=0.01,
                        type=float)
    parser.add_argument('--max_iter', default=1000,
                        type=int)
    parser.add_argument('--bridge_iter', default=100,
                        type=int)
    parser.add_argument('--diffusion_mean', default=1,
                        type=int)
    parser.add_argument('--bridge_sampling', default=0,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% Load Manifold

def load_manifold(dim:int)->None:
    
    args = parse_args()

    if args.manifold == "Euclidean":
        method = 'Local'
        M = Euclidean(N=dim)
        generator_dim = M.dim
        x0 = M.coords([0.]*dim)
        opt_val = "opt"
    elif args.manifold == "Circle":
        method = 'Embedded'
        M = S1()
        generator_dim = M.emb_dim
        x0 = M.coords([0.])
        opt_val = "gradient"
    elif args.manifold == "Sphere":
        method = 'Embedded'
        M = nSphere(N=dim)
        generator_dim = M.emb_dim
        x0 = M.coords([0.]*dim)
        opt_val = "gradient"
    elif args.manifold == "HyperbolicSpace":
        method = 'Embedded'
        M = nHyperbolicSpace(N=dim)
        generator_dim = M.emb_dim
        x0 = (jnp.concatenate((jnp.zeros(dim-1), -1.*jnp.ones(1))),)*2
        opt_val = "x0"
    elif args.manifold == "Grassmanian":
        method = 'Embedded'
        M = Grassmanian(N=2*dim,K=dim)
        generator_dim = M.emb_dim
        x0 = (jnp.eye(2*dim)[:,:dim].reshape(-1),)*2
        opt_val = "x0"
    elif args.manifold == "SO":
        method = 'Embedded'
        M = SO(N=dim)
        generator_dim = M.emb_dim
        x0 = (jnp.eye(dim).reshape(-1),)*2
        opt_val = "x0"
    elif args.manifold == "Stiefel":
        method = 'Embedded'
        M = Stiefel(N=dim, K=2)
        generator_dim = M.emb_dim
        x0 = (jnp.block([jnp.eye(2), jnp.zeros((2,dim-2))]).T.reshape(-1),)*2
        opt_val = "x0"
    elif args.manifold == "Ellipsoid":
        method = 'Embedded'
        M = nEllipsoid(N=dim, params = jnp.linspace(0.5,1.0,dim+1))
        generator_dim = M.emb_dim
        x0 = M.coords([0.]*dim)
        opt_val = "x0"
    elif args.manifold == "Cylinder":
        method = 'Embedded'
        M = Cylinder(params=(1.,jnp.array([0.,0.,1.]),jnp.pi/2.))
        generator_dim = M.dim
        x0 = M.coords([0.]*2)
        opt_val = "x0"
    elif args.manifold == "Torus":
        method = 'Embedded'
        M = Torus()        
        generator_dim = M.dim
        x0 = M.coords([0.]*2)
        opt_val = "x0"
    elif args.manifold == "Landmarks":
        method = 'Local'
        M = Landmarks(N=dim,m=2)   
        generator_dim = M.dim
        x0 = M.coords(jnp.vstack((jnp.linspace(-10.0,10.0,M.N),jnp.linspace(10.0,-10.0,M.N))).T.flatten())
        if dim >=10:
            with open('../../Data/landmarks/Papilonidae/Papilionidae_landmarks.txt', 'r') as the_file:
                all_data = [line.strip() for line in the_file.readlines()]
                
                x1 = jnp.array([float(x) for x in all_data[0].split()[2:]])
                x2 = jnp.array([float(x) for x in all_data[1].split()[2:]])
                
                idx = jnp.round(jnp.linspace(0, len(x1) - 1, args.dim)).astype(int)
                x0 = M.coords(jnp.vstack((x1[idx],x2[idx])).T.flatten())
        opt_val = "x0"
    elif args.manifold == "SPDN":
        method = 'Local'
        M = SPDN(N=dim)
        generator_dim = M.dim
        x0 = M.coords([10.]*(dim*(dim+1)//2))
        opt_val = "x0"
    elif args.manifold == "Sym":
        method = 'Local'
        M = Sym(N=dim)
        generator_dim = M.dim
        x0 = M.coords([1.]*(dim*(dim+1)//2))
        opt_val = "x0"
    elif args.manifold == "HypParaboloid":
        method = 'Local'
        M = HypParaboloid()
        generator_dim = M.dim
        x0 = M.coords([0.]*2)
        opt_val = "x0"
    else:
        return
    
    Brownian_coords(M)
    dm_bridge(M)
    
    return M, x0, method, generator_dim, opt_val

#%% Load Data

def evaluate_diffusion_mean():
    
    args = parse_args()
    
    score_mu_error = []
    bridge_mu_error = []
    score_t_error = []
    bridge_t_error = []
    for N in args.dim:
        M, x0, method, generator_dim, opt_val = load_manifold(N)
        if args.loss_type == "dsmdiagvr":
            s1_path = f"scores/{args.manifold}{N}/s1_dsmvr/"
        elif args.loss_type == "dsmdiag":
            s1_path = f"scores/{args.manifold}{N}/s1_dsm/"
        else:
            s1_path = f"scores/{args.manifold}{N}/s1_{args.loss_type}/"
        s2_path = f"scores/{args.manifold}{N}/{args.s2_type}_{args.loss_type}/"
        data_path = f"{args.data_path}{args.manifold}{N}/"
        
        if generator_dim<10:
            layers = [50,100,100,50]
        elif generator_dim<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        
        s1_state = load_model(s1_path)
        if args.s2_approx:
            s2_state = load_model(s2_path)
        else:
            s2_state = None
            
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        
        if args.s2_type == "s2":
            s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                            dim=generator_dim, r = max(generator_dim//2,1))(x))
        elif args.s2_type == "s1s2":
            @hk.transform
            def s2_model(x):
                
                s1s2 =  models.MLP_s1s2(
                    models.MLP_s1(dim=generator_dim, layers=layers), 
                    models.MLP_s2(layers_alpha=layers, 
                                  layers_beta=layers,
                                  dim=generator_dim,
                                  r = max(generator_dim//2,1))
                    )
                
                return s1s2(x)[1]
            
        ScoreEval = ScoreEvaluation(M, 
                                    s1_model=s1_model, 
                                    s1_state=s1_state, 
                                    s2_model=s2_model, 
                                    s2_state=s2_state,
                                    s2_approx=args.s2_approx, 
                                    method=method, 
                                    seed=args.seed)
        
        xs = pd.read_csv(''.join((data_path, 'xs.csv')), header=None)
        charts = pd.read_csv(''.join((data_path, 'chart.csv')), header=None)
        X_obs = (jnp.array(xs.values), jnp.array(charts.values))
        
        if opt_val == "opt":            
            mu_opt, T_opt = M.mlxt_hk(X_obs)
        elif opt_val == "gradient":
            dm_score(M, s1_model=lambda x,y,t: M.grady_log_hk(x,y,t)[0], 
                     s2_model = M.gradt_log_hk, method="Gradient")
            if args.fixed_time:
                mu_sm, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                                       step_size=args.step_size, max_iter=args.max_iter)
                T_opt = args.t0
            else:
                mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                                       step_size=args.step_size, max_iter=args.max_iter)
                T_opt = T_sm[-1]
            mu_opt = (mu_sm[0][-1], mu_sm[1][-1])
        else:
            mu_opt, T_opt = x0, args.t0
        
        
        dm_score(M, s1_model=ScoreEval.grady_log, s2_model = ScoreEval.gradt_log, method="Gradient")
        if args.fixed_time:
            mu_sm, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                                   step_size=args.step_size, max_iter=args.max_iter)
            T_sm = args.t0*jnp.ones(len(mu_sm[0]))
        else:
            mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                                   step_size=args.step_size, max_iter=args.max_iter)
            #mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]))
            
        if args.bridge_sampling:
            (thetas,chart,log_likelihood,log_likelihoods,mu_bridge) = M.diffusion_mean(X_obs,
                                                                                       num_steps=args.bridge_iter, 
                                                                                       N=1)
            
            mu_bridgex, T_bridge, mu_bridgechart = zip(*mu_bridge)
            mu_bridgex, mu_bridgechart = jnp.stack(mu_bridgex), jnp.stack(mu_bridgechart)
            T_bridge = jnp.stack(T_bridge)
        else:
            mu_bridgex, mu_bridgechart, T_bridge = [jnp.nan], [jnp.nan], [jnp.nan]
            
        print(T_sm[-1])
        if method == "Local":
            score_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_sm[0][-1]))
            score_t_error.append(jnp.linalg.norm(T_opt-T_sm[-1]))
            
            bridge_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_bridgex[-1]))
            bridge_t_error.append(jnp.linalg.norm(T_opt-T_bridge[-1]))
        else:
            score_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_sm[1][-1]))
            score_t_error.append(jnp.linalg.norm(T_opt-T_sm[-1]))
            
            bridge_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_bridgechart[-1]))
            bridge_t_error.append(jnp.linalg.norm(T_opt-T_bridge[-1]))

    score_mu_error = jnp.stack(score_mu_error)
    bridge_mu_error = jnp.stack(bridge_mu_error)
    score_t_error = jnp.stack(score_t_error)
    bridge_t_error = jnp.stack(bridge_t_error)
    
    print(score_mu_error)
    print(bridge_mu_error)
    print(score_t_error)
    print(bridge_t_error)
    
    #%%timeit [num for num in range(20)]
    
    return

#%% Main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.diffusion_mean:
        evaluate_diffusion_mean()
    else:
        pass