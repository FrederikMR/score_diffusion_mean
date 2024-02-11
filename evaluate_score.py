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
    parser.add_argument('--manifold', default="SN",
                        type=str)
    parser.add_argument('--dim', default=[2,3,5,10],
                        type=List)
    parser.add_argument('--loss_type', default="dsm",
                        type=str)
    parser.add_argument('--s2_approx', default=1,
                        type=int)
    parser.add_argument('--t', default=0.01,
                        type=float)
    parser.add_argument('--diffusion_mean', default=1,
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
    if args.manifold == "RN":
        
        for N in args.dim:
            M = Euclidean(N=N)
            Brownian_coords(M)
            dm_bridge(M)
            x0 = (jnp.zeros(N), jnp.zeros(1))
            
            file_path_s1 = 'scores/R'+str(N)+'/s1_'+ args.loss_type + '/'
            s1_state = load_model(file_path_s1)
        
            file_path_s2 = 'scores/R'+str(N)+'/s2/'
            s2_state = load_model(file_path_s2)
            if N<10:
                layers = [50,100,100,50]
            elif N<50:
                layers = [50,100,200,200,100,50]
            else:
                layers = [50,100,200,400,400,200,100,50]
            s1_model = hk.transform(lambda x: models.MLP_s1(dim=M.dim, layers=layers)(x))
            s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                            dim=M.dim, r = max(M.dim//2,1))(x))
            
            ScoreEval = ScoreEvaluation(M, s1_model=s1_model, s1_state=s1_state, s2_model=s2_model, s2_state=s2_state,
                                        s2_approx=args.s2_approx, method='Local', seed=args.seed)
                
            xs = pd.read_csv('Data/R'+str(N)+'/xs.csv', header=None)
            charts = pd.read_csv('Data/R'+str(N)+'/chart.csv', header=None)
            X_obs = (jnp.array(xs.values), jnp.array(charts.values))
            
            mu_opt, T_opt = M.mlxt_hk(X_obs)
            dm_score(M, s1_model=ScoreEval.grady_log, s2_model = ScoreEval.gradt_log, method="Gradient")
            mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([0.1]), \
                                                   step_size=0.01, max_iter=100)
        
            (thetas,chart,log_likelihood,log_likelihoods,mu_bridge) = M.diffusion_mean(X_obs,num_steps=100, N=1)
            
            mu_bridgex, T_bridge, mu_bridgechart = zip(*mu_bridge)
            mu_bridgex, _ = jnp.stack(mu_bridgex), jnp.stack(mu_bridgechart)
            T_bridge = jnp.stack(T_bridge)
            
            score_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_sm[0][-1])/N)
            score_t_error.append(jnp.linalg.norm(T_opt-T_sm[-1]))
            
            bridge_mu_error.append(jnp.linalg.norm(mu_opt[0]-mu_bridgex[-1])/N)
            bridge_t_error.append(jnp.linalg.norm(T_opt-T_bridge[-1]))
    elif args.manifold == "SN":
        for N in args.dim:
            M = nSphere(N=N)
            Brownian_coords(M)
            dm_bridge(M)
            x0 = M.coords([0.]*N)
            
            file_path_s1 = 'scores/S'+str(N)+'/s1_'+ args.loss_type + '/'
            s1_state = load_model(file_path_s1)
        
            file_path_s2 = 'scores/S'+str(N)+'/s2/'
            s2_state = load_model(file_path_s2)
            if N<10:
                layers = [50,100,100,50]
            elif N<50:
                layers = [50,100,200,200,100,50]
            else:
                layers = [50,100,200,400,400,200,100,50]
            s1_model = hk.transform(lambda x: models.MLP_s1(dim=M.emb_dim, layers=layers)(x))
            s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                            dim=M.emb_dim, r = max(M.emb_dim//2,1))(x))
            
            ScoreEval = ScoreEvaluation(M, s1_model=s1_model, s1_state=s1_state, s2_model=s2_model, s2_state=s2_state,
                                        s2_approx=args.s2_approx, method='Embedded', seed=args.seed)
                
            xs = pd.read_csv('Data/S'+str(N)+'/xs.csv', header=None)
            charts = pd.read_csv('Data/S'+str(N)+'/chart.csv', header=None)
            X_obs = (jnp.array(xs.values), jnp.array(charts.values))

            dm_score(M, s1_model=ScoreEval.grady_log, s2_model = ScoreEval.gradt_log, method="Gradient")
            mu_sm, T_sm, gradx_sm, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([0.1]), \
                                                   step_size=0.01, max_iter=100)
            #diffusion_mean.initialize(M, s1_model=lambda x,y,t: M.grady_log_hk(x,y,t)[0], s2_model = lambda x,y,t: jacfwdx(lambda y: M.grady_log_hk(x,y,t)[0])(y), method="Gradient")
            dm_score(M, s1_model=lambda x,y,t: M.grady_log_hk(x,y,t)[0], s2_model = M.gradt_log_hk, method="Gradient")
            mu_opt, T_opt, _, _ = M.sm_dmxt(X_obs, (X_obs[0][0], X_obs[1][0]), jnp.array([0.1]))
        
            (thetas,chart,log_likelihood,log_likelihoods,mu_bridge) = M.diffusion_mean(X_obs,num_steps=100, N=1)
            
            mu_bridgex, T_bridge, mu_bridgechart = zip(*mu_bridge)
            mu_bridgex, mu_bridgechart = jnp.stack(mu_bridgex), jnp.stack(mu_bridgechart)
            T_bridge = jnp.stack(T_bridge)
            
            score_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_sm[1][-1])/N)
            score_t_error.append(jnp.linalg.norm(T_opt-T_sm[-1]))
            
            bridge_mu_error.append(jnp.linalg.norm(mu_opt[1]-mu_bridgechart[-1])/N)
            bridge_t_error.append(jnp.linalg.norm(T_opt-T_bridge[-1]))
            
    score_mu_error = jnp.stack(score_mu_error)
    bridge_mu_error = jnp.stack(bridge_mu_error)
    score_t_error = jnp.stack(score_t_error)
    bridge_t_error = jnp.stack(bridge_t_error)
    
    return

#%% Main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.diffusion_mean:
        evaluate_diffusion_mean()
    else:
        pass