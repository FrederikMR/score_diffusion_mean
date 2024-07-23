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
from jax import lax

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
    parser.add_argument('--manifold', default="Sphere",
                        type=str)
    parser.add_argument('--dim', default=[2,3,5,10],
                        type=List)
    parser.add_argument('--s1_loss_type', default="dsm",
                        type=str)
    parser.add_argument('--s2_loss_type', default="dsm",
                        type=str)
    parser.add_argument('--dt_approx', default="s1",
                        type=str)
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
    parser.add_argument('--benchmark', default=0,
                        type=int)
    parser.add_argument('--method', default="Gradient",
                        type=str)
    parser.add_argument('--data_path', default='data/',
                        type=str)
    parser.add_argument('--save_path', default='table/estimates/',
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
    dt_ntrain = []
    score_mu_time = []
    score_std_time = []
    bridge_mu_time = []
    bridge_std_time = []
    for N in args.dim:
        
        M, x0, method, generator_dim, layers, opt_val = load_manifold(args.manifold,
                                                                               N)
        layers_s1, layers_s2 = layers

        if method == "LocalSampling":
            method = "Local"
        else:
            method = "Embedded"
            
        Brownian_coords(M)
        
        if args.manifold == "Landmarks":
            phi = "Landmarks"
        else:
            phi = None
            
        dm_bridge(M, phi)
        
        st_path = f"{args.score_path}{args.manifold}{N}/st/"
        s1_path = f"{args.score_path}{args.manifold}{N}/s1_{args.s1_loss_type}/"
        s2_path = f"{args.score_path}{args.manifold}{N}/{args.dt_approx}_{args.s2_loss_type}/"
        data_path = f"{args.data_path}{args.manifold}{N}/"

        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers_s1)(x))
        st_model = hk.transform(lambda x: models.MLP_t(dim=generator_dim, layers=layers_s1)(x))
        if "diag" in args.s2_loss_type:
            s2_model = hk.transform(lambda x: models.MLP_diags2(layers_alpha=layers_s2, layers_beta=layers_s2,
                                                                dim=generator_dim, 
                                                                r = max((generator_dim-1)//2,1))(x))
        else:
            s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers_s2, layers_beta=layers_s2,
                                                            dim=generator_dim, 
                                                            r = max((generator_dim-1)//2,1))(x))
        
        if "s1s2" in args.dt_approx:
            s1_path = s2_path
            if "diag" in args.s2_loss_type:
                @hk.transform
                def s1_model(x):
                    
                    s1s2 =  models.MLP_diags1s2(
                        models.MLP_s1(dim=generator_dim, layers=layers_s1), 
                        models.MLP_s2(layers_alpha=layers_s2, 
                                      layers_beta=layers_s2,
                                      dim=generator_dim,
                                      r = max((generator_dim-1)//2,1))
                        )
                    
                    return s1s2(x)[0]
                
                @hk.transform
                def s2_model(x):
                    
                    s1s2 =  models.MLP_diags1s2(
                        models.MLP_s1(dim=generator_dim, layers=layers_s1), 
                        models.MLP_s2(layers_alpha=layers_s2, 
                                      layers_beta=layers_s2,
                                      dim=generator_dim,
                                      r = max((generator_dim-1)//2,1))
                        )
                    
                    return s1s2(x)[1]
            else:    
                @hk.transform
                def s1_model(x):
                    
                    s1s2 =  models.MLP_diags1s2(
                        models.MLP_s1(dim=generator_dim, layers=layers_s1), 
                        models.MLP_s2(layers_alpha=layers_s2, 
                                      layers_beta=layers_s2,
                                      dim=generator_dim,
                                      r = max((generator_dim-1)//2,1))
                        )
                    
                    return s1s2(x)[0]
                
                @hk.transform
                def s2_model(x):
                    
                    s1s2 =  models.MLP_diags1s2(
                        models.MLP_s1(dim=generator_dim, layers=layers_s1), 
                        models.MLP_s2(layers_alpha=layers_s2, 
                                      layers_beta=layers_s2,
                                      dim=generator_dim,
                                      r = max((generator_dim-1)//2,1))
                        )
                    
                    return s1s2(x)[1]
        
        s1_state = load_model(s1_path)
        s1_ntrain.append(len(jnp.load(''.join((s1_path, 'loss_arrays.npy')))))
        rng_key = jrandom.PRNGKey(args.seed)
        
        s1_fun = lambda x,y,t: s1_model.apply(s1_state.params, rng_key, jnp.hstack((x,y,t)))
        if args.dt_approx == "s1":
            s2_state = None
            dt_ntrain.append(jnp.nan)
            st_fun = None
            s2_fun = None
        elif "s2" in args.dt_approx:
            s2_state = load_model(s2_path)
            dt_ntrain.append(len(jnp.load(''.join((s2_path, 'loss_arrays.npy')))))
            st_fun = None
            s2_fun = lambda x,y,t: lax.stop_gradient(s2_model.apply(s2_state.params, rng_key, jnp.hstack((x,y,t))))
        elif args.dt_approx == "dt":
            st_state = load_model(st_path)
            dt_ntrain.append(len(jnp.load(''.join((st_path, 'loss_arrays.npy')))))
            st_fun = lambda x,y,t: st_model.apply(st_state.params, rng_key, jnp.hstack((x,y,t)))
            s2_fun = None

        xs = pd.read_csv(''.join((data_path, 'xs.csv')), header=None)
        charts = pd.read_csv(''.join((data_path, 'chart.csv')), header=None)
        X_obs = (jnp.array(xs.values), jnp.array(charts.values))
        if opt_val == "opt":            
            mu_opt, T_opt = M.mlxt_hk(X_obs)
        elif opt_val == "gradient":
            dm_score(M, s1_model=lambda x,y,t: M.grady_log_hk(x,y,t)[0], 
                     s2_model = M.gradt_log_hk, method=args.method)
            mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t_init]), \
                                                   step_size=args.step_size, max_iter=args.score_iter)
            T_opt = T_sm[-1]
            mu_opt = (mu_sm[0][-1], mu_sm[1][-1])
        else:
            mu_opt, T_opt = x0, 0.5

        ScoreEval = ScoreEvaluation(M,
                                    s1_model=s1_fun,
                                    s2_model=s2_fun,
                                    st_model=st_fun,
                                    method=method, 
                                    )

        dm_score(M, 
                 s1_model=ScoreEval.grady_log,
                 s2_model = ScoreEval.gradt_log, method=args.method)
        mu_sm, T_sm, gradx_sm, gradt_sm = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t_init]), \
                                               step_size=args.step_size, max_iter=args.score_iter)
        time_fun = lambda x: M.sm_dmxt(X_obs, (x[0], x[1]), jnp.array([args.t_init]), step_size=args.step_size, 
                                       max_iter=5)
        time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                             number=1, globals=locals(), repeat=args.timing_repeats)
        score_mu_time.append(jnp.mean(jnp.array(time)))
        score_std_time.append(jnp.std(jnp.array(time)))
            #mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t_init]))
        print(s1_ntrain)
        print(T_opt)
        print(T_sm[-1])
        print(mu_opt[1])
        print(mu_sm[1][-1])
        print(mu_opt[0])
        print(mu_sm[0][-1])
        if args.benchmark:
            X_obs = (X_obs[0].astype(jnp.float64), X_obs[1].astype(jnp.float64))
            (thetas,chart,log_likelihood,log_likelihoods,mu_bridge) = M.diffusion_mean(X_obs,
                                                                                       num_steps=args.bridge_iter, 
                                                                                       N=1)
            
            mu_bridgex, T_bridge, mu_bridgechart = zip(*mu_bridge)
            mu_bridgex, mu_bridgechart = jnp.stack(mu_bridgex), jnp.stack(mu_bridgechart)
            T_bridge = jnp.stack(T_bridge)
            (thetas,chart,log_likelihood,log_likelihoods,mu_bridge) = M.diffusion_mean(X_obs,
                                                                                       num_steps=5, 
                                                                                       N=1)
            time = timeit.repeat('M.diffusion_mean(X_obs,num_steps=5,N=1)', number=1,
                                 globals=locals(), repeat=args.timing_repeats)
            bridge_mu_time.append(jnp.mean(jnp.array(time)))
            bridge_std_time.append(jnp.std(jnp.array(time)))
        else:
            mu_bridgex, mu_bridgechart, T_bridge = [jnp.nan], [jnp.nan], [jnp.nan]
            bridge_mu_time.append(jnp.nan)
            bridge_std_time.append(jnp.nan)
            
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

    print(mu_bridgex[-1])
    print(mu_bridgechart[-1])
    print(T_bridge[-1])
    print(score_mu_error)
    print(score_t_error)
    error = {'score_mu_error': jnp.stack(score_mu_error),
             'bridge_mu_error': jnp.stack(bridge_mu_error),
             'score_t_error': jnp.stack(score_t_error),
             'bridge_t_error': jnp.stack(bridge_t_error),
             'dim': args.dim,
             's1_ntrain': jnp.stack(s1_ntrain),
             'dt_ntrain': jnp.stack(dt_ntrain),
             'score_mu_time': jnp.stack(score_mu_time),
             'score_std_time': jnp.stack(score_std_time),
             'bridge_mu_time': jnp.stack(bridge_mu_time),
             'bridge_std_time': jnp.stack(bridge_std_time)
             }
    
    save_path = f"{args.save_path}timing_{args.manifold}.pkl"
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
    diff_error = []
    s1_ntrain = []
    score_mu_time = []
    score_std_time = []
    frechet_mu_time = []
    frechet_std_time = []
    for N in args.dim:
        M, x0, method, generator_dim, layers, opt_val = load_manifold(args.manifold,
                                                                               N)
        layers_s1, layers_s2 = layers
        
        s1_path = f"scores/{args.manifold}{N}/s1T_{args.s1_loss_type}/"
        s2_path = f"scores/{args.manifold}{N}/{args.dt_approx}_{args.s2_loss_type}/"
        data_path = f"{args.data_path}{args.manifold}{N}/"
        
        if method == "LocalSampling":
            method = "Local"
        else:
            method = "Embedded"
        
        
        s1_state = load_model(s1_path)
        s1_ntrain.append(len(jnp.load(''.join((s1_path, 'loss_arrays.npy')))))
            
        rng_key = jrandom.PRNGKey(args.seed)
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers_s1)(x))
        s1_fun = lambda x,y,t: s1_model.apply(s1_state.params, rng_key, jnp.hstack((x,y,t)))
        
            
        ScoreEval = ScoreEvaluation(M, 
                                    s1_model=s1_fun,
                                    s2_model = None,
                                    st_model = None,
                                    method = method,
                                    )
        
        xs = pd.read_csv(''.join((data_path, 'xs.csv')), header=None)
        charts = pd.read_csv(''.join((data_path, 'chart.csv')), header=None)
        X_obs = (jnp.array(xs.values), jnp.array(charts.values))
        
        if opt_val == "opt":            
            mu_opt, T_opt = M.mlxt_hk(X_obs)
        else:
            mu_opt = x0
        
        def s1_evaluate(x,y,t):
            
            val1 = ScoreEval.grady_log(x,y,t)
            
            return val1/jnp.linalg.norm(val1)
            
        dm_score(M, s1_model=s1_evaluate, 
                 s2_model = ScoreEval.gradt_log, method="Gradient")
        mu_sm, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                               step_size=args.step_size, max_iter=args.score_iter)
        time_fun = lambda x: M.sm_dmx(X_obs, (x[0], x[1]), jnp.array([args.t0]), 
                                      step_size=args.step_size, 
                                      max_iter=5)
        time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                             number=1, globals=locals(), repeat=args.timing_repeats)
        score_mu_time.append(jnp.mean(jnp.array(time)))
        score_std_time.append(jnp.std(jnp.array(time)))
        
        print(mu_opt[1])
        print(mu_sm[1][-1])
        print(mu_opt[0])
        print(mu_sm[0][-1])
        
        if args.benchmark:
            Frechet_mean(M)
            #mu_frechet,loss,iterations,vs = M.Frechet_mean(zip(X_obs[0], X_obs[1]),(X_obs[0][0], X_obs[1][0]),
            #                                               options={'num_steps':args.bridge_iter})
            #time_fun = lambda x: M.Frechet_mean(zip(X_obs[0], X_obs[1]),(x[0], x[1]), 
            #                                    options={'num_steps':10})
            #print(mu_frechet)
            #mu_frechet,loss,iterations,vs = M.Frechet_mean(zip(X_obs[0], X_obs[1]),(X_obs[0][0], X_obs[1][0]),
            #                                               options={'num_steps':10})
            #time_fun = lambda x: M.Frechet_mean(zip(X_obs[0], X_obs[1]),(x[0], x[1]), 
            #                                    options={'num_steps':10})            
            #print(mu_frechet)
            dm_score(M, s1_model=lambda x,y,t: M.Log(y,x[1]), 
                     s2_model = ScoreEval.gradt_log, method="Gradient")
            mu_frechet, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
                                                       step_size=args.step_size, max_iter=args.bridge_iter)
            mu_frechet = (mu_frechet[0][-1], mu_frechet[1][-1])
            time_fun = lambda x: M.sm_dmx(X_obs, (x[0], x[1]), jnp.array([args.t0]), 
                                              step_size=args.step_size, 
                                              max_iter=5)
            print(mu_frechet)
            #if not (args.manifold == "Sphere" or args.manifold=="Ellipsoid"):
            #    mu_frechet,loss,iterations,vs = M.Frechet_mean(zip(X_obs[0], X_obs[1]),(X_obs[0][0], X_obs[1][0]))
            #    time_fun = lambda x: M.Frechet_mean(zip(X_obs[0], X_obs[1]),(x[0], x[1]), 
            #                                        options={'num_steps':10})
            #    print(mu_frechet)
            #else:
            #    mu_frechet, _ = M.sm_dmx(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([args.t0]), \
            #                                           step_size=args.step_size, max_iter=args.score_iter)
            #    mu_frechet = (mu_frechet[0][-1], mu_frechet[1][-1])
            #    time_fun = lambda x: M.sm_dmx(X_obs, (x[0], x[1]), jnp.array([args.t0]), 
            #                                  step_size=args.step_size, 
            #                                  max_iter=10)
            #    print(mu_frechet)
            time = timeit.repeat('time_fun((X_obs[0][0], X_obs[1][0]))',
                                 number=1, globals=locals(), repeat=args.timing_repeats)
            frechet_mu_time.append(jnp.mean(jnp.array(time)))
            frechet_std_time.append(jnp.std(jnp.array(time)))
        else:
            mu_frechet = (jnp.nan, jnp.nan)
            frechet_mu_time.append(jnp.nan)
            frechet_std_time.append(jnp.nan)
            
        if method == "Local":
            diff_error.append(jnp.linalg.norm(mu_frechet[0]-mu_sm[0][-1]))
            score_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_sm[0][-1]))
            frechet_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_frechet[0]))
        else:
            diff_error.append(jnp.linalg.norm(mu_frechet[1]-mu_sm[1][-1]))  
            score_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_sm[1][-1]))
            frechet_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_frechet[1]))
            
    print(score_mu_error)

    error = {'diff_error': jnp.stack(diff_error),
             'score_mu_error': jnp.stack(score_mu_error),
             'frechet_mu_error': jnp.stack(frechet_mu_error),
             'dim': args.dim,
             's1_ntrain': jnp.stack(s1_ntrain),
             'score_mu_time': jnp.stack(score_mu_time),
             'score_std_time': jnp.stack(score_std_time),
             'frechet_mu_time': jnp.stack(frechet_mu_time),
             'frechet_std_time': jnp.stack(frechet_std_time)
             }
    
    save_path = f"{args.save_path}timing_frechet_{args.manifold}.pkl"
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
