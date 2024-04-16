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
import jax.random as jrandom

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

from load_manifold import load_manifold

#jaxgeometry
from jaxgeometry.manifolds import *
from jaxgeometry.stochastics import Brownian_coords
from jaxgeometry.statistics.score_matching.model_loader import load_model
from jaxgeometry.statistics.score_matching import diffusion_mean as dm_score
from jaxgeometry.statistics.score_matching import ScoreEvaluation
from jaxgeometry.statistics import diffusion_mean as dm_bridge
from jaxgeometry.statistics import Frechet_mean

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="HypParaboloid",
                        type=str)
    parser.add_argument('--dim', default=[2],
                        type=List)
    parser.add_argument('--s1_loss_type', default="dsmvr",
                        type=str)
    parser.add_argument('--s2_loss_type', default="dsmvr",
                        type=str)
    parser.add_argument('--s2_type', default="s2",
                        type=str)
    parser.add_argument('--s2_approx', default=1,
                        type=int)
    parser.add_argument('--fixed_t', default=0,
                        type=int)
    parser.add_argument('--t0', default=0.01,
                        type=float)
    parser.add_argument('--step_size', default=0.01,
                        type=float)
    parser.add_argument('--score_iter', default=1000,
                        type=int)
    parser.add_argument('--bridge_iter', default=100,
                        type=int)
    parser.add_argument('--t_init', default=0.2,
                        type=float)
    parser.add_argument('--estimate', default="diffusion_mean",
                        type=str)
    parser.add_argument('--bridge_sampling', default=0,
                        type=int)
    parser.add_argument('--method', default="Gradient",
                        type=str)
    parser.add_argument('--data_path', default='../data/',
                        type=str)
    parser.add_argument('--save_path', default='../results/estimates/',
                        type=str)
    parser.add_argument('--score_path', default='scores/',
                        type=str)
    parser.add_argument('--timing_repeats', default=5,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

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
        
        M, x0, method, generator_dim, layers, opt_val = load_manifold(args.manifold,
                                                                               N)
        if not (method == "Local"):
            method = "Embedded"
            
        Brownian_coords(M)
        dm_bridge(M)
        
        s1_path = f"{args.score_path}{args.manifold}{N}/s1_{args.s1_loss_type}/"
        s2_path = f"{args.score_path}{args.manifold}{N}/{args.s2_type}_{args.s2_loss_type}/"
        data_path = f"{args.data_path}{args.manifold}{N}/"
        
        s1_state = load_model(s1_path)
        s1_ntrain.append(len(jnp.load(''.join((s1_path, 'loss_arrays.npy')))))
        if args.s2_approx:
            s2_state = load_model(s2_path)
            s2_ntrain.append(len(jnp.load(''.join((s2_path, 'loss_arrays.npy')))))
        else:
            s2_state = None
            s2_ntrain.append(jnp.nan)

        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        if "diag" in args.s2_loss_type:
            s2_model = hk.transform(lambda x: models.MLP_diags2(layers_alpha=layers, layers_beta=layers,
                                                            dim=generator_dim, 
                                                            r = max(generator_dim//2,1))(x))
        
            @hk.transform
            def s1s2_model(x):
                
                s1s2 =  models.MLP_diags1s2(
                    models.MLP_s1(dim=generator_dim, layers=layers), 
                    models.MLP_s2(layers_alpha=layers, 
                                  layers_beta=layers,
                                  dim=generator_dim,
                                  r = max(generator_dim//2,1))
                    )
                
                return s1s2(x)
        else:    
            s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                            dim=generator_dim, 
                                                            r = max(generator_dim//2,1))(x))
        
            @hk.transform
            def s1s2_model(x):
                
                s1s2 =  models.MLP_s1s2(
                    models.MLP_s1(dim=generator_dim, layers=layers), 
                    models.MLP_s2(layers_alpha=layers, 
                                  layers_beta=layers,
                                  dim=generator_dim,
                                  r = max(generator_dim//2,1))
                    )
                
                return s1s2(x)

        xs = pd.read_csv(''.join((data_path, 'xs.csv')), header=None)
        charts = pd.read_csv(''.join((data_path, 'chart.csv')), header=None)
        X_obs = (jnp.array(xs.values), jnp.array(charts.values))
        
        if opt_val == "opt":            
            mu_opt, T_opt = M.mlxt_hk(X_obs)
        elif opt_val == "gradient":
            dm_score(M, s1_model=lambda x,y,t: M.grady_log_hk(x,y,t)[0], 
                     s2_model = M.gradt_log_hk, method=args.method)
            if args.fixed_t:
                mu_sm, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t_init]), \
                                                       step_size=args.step_size, max_iter=args.score_iter)
                T_opt = args.t_init
            else:
                mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t_init]), \
                                                       step_size=args.step_size, max_iter=args.score_iter)
                T_opt = T_sm[-1]
            mu_opt = (mu_sm[0][-1], mu_sm[1][-1])
        else:
            mu_opt, T_opt = x0, 0.5

        rng_key = jrandom.PRNGKey(args.seed)
        s1_fun = lambda x,y,t: s1_model.apply(s1_state.params, rng_key, jnp.hstack((x,y,t)))
        if args.s2_approx:
            s2_fun = lambda x,y,t: s2_model.apply(s2_state.params, rng_key, jnp.hstack((x,y,t)))
        else:
            s2_fun = None

        ScoreEval = ScoreEvaluation(M,
                                    s1_model=s1_fun, 
                                    s2_model=s2_fun,#s2_model_test2, 
                                    method=method, 
                                    )

        dm_score(M, 
                 s1_model=ScoreEval.grady_log,
                 s2_model = ScoreEval.gradt_log, method=args.method)
        if args.fixed_t:
            mu_sm, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t_init]), \
                                                   step_size=args.step_size, max_iter=args.score_iter)
            T_sm = args.t_init*jnp.ones(len(mu_sm[0]))
            time_fun = lambda x: M.sm_dmx(X_obs, (x[0], x[1]), jnp.array([args.t_init]), step_size=args.step_size, max_iter=args.bridge_iter)
            time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                                 number=1, globals=locals(), repeat=args.timing_repeats)
            score_mu_time.append(jnp.mean(jnp.array(time)))
            score_std_time.append(jnp.std(jnp.array(time)))
        else:
            mu_sm, T_sm, gradx_sm, gradt_sm = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t_init]), \
                                                   step_size=args.step_size, max_iter=args.score_iter)
            time_fun = lambda x: M.sm_dmxt(X_obs, (x[0], x[1]), jnp.array([args.t_init]), step_size=args.step_size, max_iter=args.bridge_iter)
            time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                                 number=1, globals=locals(), repeat=args.timing_repeats)
            score_mu_time.append(jnp.mean(jnp.array(time)))
            score_std_time.append(jnp.std(jnp.array(time)))
            #mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t_init]))
        print(T_opt)
        print(T_sm[-1])
        print(mu_opt[1])
        print(mu_sm[1][-1])
        if args.bridge_sampling:
            (thetas,chart,log_likelihood,log_likelihoods,mu_bridge) = M.diffusion_mean(X_obs,
                                                                                       num_steps=args.bridge_iter, 
                                                                                       N=1)
            
            mu_bridgex, T_bridge, mu_bridgechart = zip(*mu_bridge)
            mu_bridgex, mu_bridgechart = jnp.stack(mu_bridgex), jnp.stack(mu_bridgechart)
            T_bridge = jnp.stack(T_bridge)
            time = timeit.repeat('M.diffusion_mean(X_obs,num_steps=args.bridge_iter,N=1)', number=1,
                                 globals=locals(), repeat=args.timing_repeats)
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
    print(score_t_error)
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
        s1_path = f"../scores/{args.manifold}{N}/s1T_{args.s1_loss_type}/"
        s2_path = f"../scores/{args.manifold}{N}/{args.s2_type}_{args.s2_loss_type}/"
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
        mu_sm, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                               step_size=args.step_size, max_iter=args.score_iter)
        time_fun = lambda x: M.sm_dmx(X_obs, (x[0], x[1]), jnp.array([args.t0]), step_size=args.step_size, max_iter=args.bridge_iter)
        time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                             number=1, globals=locals(), repeat=args.timing_repeats)
        score_mu_time.append(jnp.mean(jnp.array(time)))
        score_std_time.append(jnp.std(jnp.array(time)))
        
        if args.bridge_sampling:
            Frechet_mean(M)
            mu_frechet,loss,iterations,vs = M.Frechet_mean(zip(X_obs[0], X_obs[1]),(X_obs[0][0], X_obs[1][0]))
            time_fun = lambda x: M.Frechet_mean(zip(X_obs[0], X_obs[1]),(x[0], x[1]))
            time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                                 number=1, globals=locals(), repeat=args.timing_repeats)
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
    
    if args.estimate == "diffusion_mean":
        evaluate_diffusion_mean()
    else:
        evaluate_frechet_mean()