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

#pickle
import pickle

#os
import os

from typing import List

import timeit

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
from jaxgeometry.statistics import Frechet_mean
from jaxgeometry.statistics import Frechet_mean

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="Sym",
                        type=str)
    parser.add_argument('--dim', default=[20],
                        type=List)
    parser.add_argument('--loss_type', default="dsmvr",
                        type=str)
    parser.add_argument('--s2_approx', default=0,
                        type=int)
    parser.add_argument('--fixed_time', default=0,
                        type=int)
    parser.add_argument('--s2_type', default="s2",
                        type=str)
    parser.add_argument('--method', default="JAX",
                        type=str)
    parser.add_argument('--data_path', default='../data/',
                        type=str)
    parser.add_argument('--save_path', default='../results/estimates/',
                        type=str)
    parser.add_argument('--score_path', default='scores/',
                        type=str)
    parser.add_argument('--t', default=0.1,
                        type=float)
    parser.add_argument('--t0', default=0.2,
                        type=float)
    parser.add_argument('--step_size', default=0.01,
                        type=float)
    parser.add_argument('--max_iter', default=1000,
                        type=int)
    parser.add_argument('--repeats', default=5,
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
        x0 = M.coords(jnp.vstack((jnp.linspace(-5.0,0.0,M.N),jnp.linspace(5.0,0.0,M.N))).T.flatten())
        if dim >=10:
            with open('../../Data/landmarks/Papilonidae/Papilionidae_landmarks.txt', 'r') as the_file:
                all_data = [line.strip() for line in the_file.readlines()]
                
                x1 = jnp.array([float(x) for x in all_data[0].split()[2:]])
                x2 = jnp.array([float(x) for x in all_data[1].split()[2:]])
                
                idx = jnp.round(jnp.linspace(0, len(x1) - 1, dim)).astype(int)
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
    elif args.manifold == 'gp_mnist':
        
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
    s1_ntrain = []
    s2_ntrain = []
    score_mu_time = []
    score_std_time = []
    bridge_mu_time = []
    bridge_std_time = []
    for N in args.dim:
        M, x0, method, generator_dim, opt_val = load_manifold(N)
        if args.loss_type == "dsmdiagvr":
            s1_path = f"{args.score_path}{args.manifold}{N}/s1_dsmvr/"
        elif args.loss_type == "dsmdiag":
            s1_path = f"{args.score_path}{args.manifold}{N}/s1_dsm/"
        else:
            s1_path = f"{args.score_path}{args.manifold}{N}/s1_{args.loss_type}/"
        s2_path = f"{args.score_path}{args.manifold}{N}/{args.s2_type}_{args.loss_type}/"
        data_path = f"{args.data_path}{args.manifold}{N}/"

        if generator_dim<10:
            layers = [50,100,100,50]
        elif generator_dim<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        
        s1_state = load_model(s1_path)
        s1_ntrain.append(len(jnp.load(''.join((s1_path, 'loss_arrays.npy')))))
        if args.s2_approx:
            s2_state = load_model(s2_path)
            s2_ntrain.append(len(jnp.load(''.join((s2_path, 'loss_arrays.npy')))))
        else:
            s2_state = None
            s2_ntrain.append(jnp.nan)
            
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
            
           # s1_state = s2_state
           # @hk.transform
           # def s1_model(x):
           #     
           #     s1s2 =  models.MLP_s1s2(
           #         models.MLP_s1(dim=generator_dim, layers=layers), 
           #         models.MLP_s2(layers_alpha=layers, 
           #                       layers_beta=layers,
           #                       dim=generator_dim,
           #                       r = max(generator_dim//2,1))
           #         )
           #     
           #     return s1s2(x)[0]
            
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
                     s2_model = M.gradt_log_hk, method=args.method)
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
            mu_opt, T_opt = x0, 0.5
        
        
        dm_score(M, s1_model=ScoreEval.grady_log, s2_model = ScoreEval.gradt_log, method=args.method)
        if args.fixed_time:
            mu_sm, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                                   step_size=args.step_size, max_iter=args.max_iter)
            T_sm = args.t0*jnp.ones(len(mu_sm[0]))
            time_fun = lambda x: M.sm_dmx(X_obs, (x[0], x[1]), jnp.array([args.t0]), step_size=args.step_size, max_iter=args.bridge_iter)
            time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                                 number=1, globals=locals(), repeat=args.repeats)
            score_mu_time.append(jnp.mean(jnp.array(time)))
            score_std_time.append(jnp.std(jnp.array(time)))
        else:
            mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                                   step_size=args.step_size, max_iter=args.max_iter)
            print(T_sm[-1])
            print(mu_sm[0][-1])
            print(mu_sm[1][-1])
            time_fun = lambda x: M.sm_dmxt(X_obs, (x[0], x[1]), jnp.array([args.t0]), step_size=args.step_size, max_iter=args.bridge_iter)
            time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                                 number=1, globals=locals(), repeat=args.repeats)
            score_mu_time.append(jnp.mean(jnp.array(time)))
            score_std_time.append(jnp.std(jnp.array(time)))
            #mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]))
            
        if args.bridge_sampling:
            (thetas,chart,log_likelihood,log_likelihoods,mu_bridge) = M.diffusion_mean(X_obs,
                                                                                       num_steps=args.bridge_iter, 
                                                                                       N=1)
            
            mu_bridgex, T_bridge, mu_bridgechart = zip(*mu_bridge)
            mu_bridgex, mu_bridgechart = jnp.stack(mu_bridgex), jnp.stack(mu_bridgechart)
            T_bridge = jnp.stack(T_bridge)
            time = timeit.repeat('M.diffusion_mean(X_obs,num_steps=args.bridge_iter,N=1)', number=1,
                                 globals=locals(), repeat=args.repeats)
            bridge_mu_time.append(jnp.mean(jnp.array(time)))
            bridge_std_time.append(jnp.std(jnp.array(time)))
        else:
            mu_bridgex, mu_bridgechart, T_bridge = [jnp.nan], [jnp.nan], [jnp.nan]
            bridge_mu_time.append(jnp.nan)
            bridge_std_time.append(jnp.nan)
            
        if method == "Local":
            D = len(mu_opt[0])
            score_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_sm[0][-1])/D)
            score_t_error.append(jnp.linalg.norm(T_opt-T_sm[-1]))
            
            bridge_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_bridgex[-1])/D)
            bridge_t_error.append(jnp.linalg.norm(T_opt-T_bridge[-1]))
        else:
            D = len(mu_opt[1])
            score_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_sm[1][-1])/D)
            score_t_error.append(jnp.linalg.norm(T_opt-T_sm[-1]))
            
            bridge_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_bridgechart[-1])/D)
            bridge_t_error.append(jnp.linalg.norm(T_opt-T_bridge[-1]))
    print(score_mu_error)
    error = {'score_mu_error': jnp.stack(score_mu_error),
             'bridge_mu_error': jnp.stack(bridge_mu_error),
             'score_t_error': jnp.stack(score_t_error),
             'bridge_t_error': jnp.stack(bridge_t_error),
             'dim': args.dim,
             's1_ntrain': jnp.stack(s1_ntrain),
             's2_ntrain': jnp.stack(s2_ntrain),
             'score_mu_time': jnp.stack(score_mu_time),
             'score_std_time': jnp.stack(score_std_time),
             'bridge_mu_time': jnp.stack(bridge_mu_time),
             'bridge_std_time': jnp.stack(bridge_std_time)
             }
    
    save_path = f"{args.save_path}{args.manifold}.pkl"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(error, f)
    
    return

#%% Load Data

def evaluate_frechet_mean():
    
    args = parse_args()
    
    score_mu_error = []
    frechet_mu_error = []
    s1_ntrain = []
    s2_ntrain = []
    score_mu_time = []
    score_std_time = []
    frechet_mu_time = []
    frechet_std_time = []
    for N in args.dim:
        M, x0, method, generator_dim, opt_val = load_manifold(N)
        if args.loss_type == "dsmdiagvr":
            s1_path = f"../scores/{args.manifold}{N}/s1T_dsmvr/"
        elif args.loss_type == "dsmdiag":
            s1_path = f"../scores/{args.manifold}{N}/s1T_dsm/"
        else:
            s1_path = f"../scores/{args.manifold}{N}/s1T_{args.loss_type}/"
        s2_path = f"../scores/{args.manifold}{N}/{args.s2_type}_{args.loss_type}/"
        data_path = f"{args.data_path}{args.manifold}{N}/"

        if generator_dim<10:
            layers = [50,100,100,50]
        elif generator_dim<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        
        s1_state = load_model(s1_path)
        s1_ntrain.append(len(jnp.load(''.join((s1_path, 'loss_arrays.npy')))))
        if args.s2_approx:
            s2_state = load_model(s2_path)
            s2_ntrain.append(len(jnp.load(''.join((s2_path, 'loss_arrays.npy')))))
        else:
            s2_state = None
            s2_ntrain.append(jnp.nan)
            
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
        else:
            mu_opt = x0
        
        
        dm_score(M, s1_model=lambda x,y,t: t*ScoreEval.grady_log(x,y,t), 
                 s2_model = ScoreEval.gradt_log, method="Gradient")
        mu_sm, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t]), \
                                               step_size=args.step_size, max_iter=args.max_iter)
        time_fun = lambda x: M.sm_dmx(X_obs, (x[0], x[1]), jnp.array([args.t]), step_size=args.step_size, max_iter=args.bridge_iter)
        time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                             number=1, globals=locals(), repeat=args.repeats)
        score_mu_time.append(jnp.mean(jnp.array(time)))
        score_std_time.append(jnp.std(jnp.array(time)))
        
        if args.bridge_sampling:
            Frechet_mean(M)
            mu_frechet,loss,iterations,vs = M.Frechet_mean(zip(X_obs[0], X_obs[1]),(X_obs[0][0], X_obs[1][0]))
            time_fun = lambda x: M.Frechet_mean(zip(X_obs[0], X_obs[1]),(x[0], x[1]))
            time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                                 number=1, globals=locals(), repeat=args.repeats)
            frechet_mu_time.append(jnp.mean(jnp.array(time)))
            frechet_std_time.append(jnp.std(jnp.array(time)))
        else:
            mu_frechet = (jnp.nan, jnp.nan)
            frechet_mu_time.append(jnp.nan)
            frechet_std_time.append(jnp.nan)
            
        if method == "Local":
            D = len(mu_opt[0])
            score_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_sm[0][-1])/D)
            frechet_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_frechet[0])/D)
        else:
            D = len(mu_opt[1])
            score_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_sm[1][-1])/D)            
            frechet_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_frechet[1])/D)

    error = {'score_mu_error': jnp.stack(score_mu_error),
             'frechet_mu_error': jnp.stack(frechet_mu_error),
             'dim': args.dim,
             's1_ntrain': jnp.stack(s1_ntrain),
             's2_ntrain': jnp.stack(s2_ntrain),
             'score_mu_time': jnp.stack(score_mu_time),
             'score_std_time': jnp.stack(score_std_time),
             'frechet_mu_time': jnp.stack(frechet_mu_time),
             'frechet_std_time': jnp.stack(frechet_std_time)
             }
    
    save_path = f"{args.save_path}frechet_{args.manifold}.pkl"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(error, f)
    
    return

#%% Main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.diffusion_mean:
        evaluate_diffusion_mean()
    else:
        evaluate_frechet_mean()