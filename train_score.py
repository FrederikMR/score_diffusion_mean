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
import jax.random as jran
from jax import Array

from scipy import ndimage

from gp.gp import RM_EG

#haiku
import haiku as hk

#argparse
import argparse

#scores
from models import models

from typing import NamedTuple

#os
import os

from load_manifold import load_manifold

#jaxgeometry
from jaxgeometry.manifolds import *
from jaxgeometry.statistics.score_matching import train_s1, train_s2, train_s1s2, RiemannianBrownianGenerator
from jaxgeometry.statistics.score_matching.model_loader import load_model
from ManLearn.train_MNIST import load_dataset as load_mnist

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="Euclidean",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--train_net', default="s1",
                        type=str)
    parser.add_argument('--s1_loss_type', default="dsmvr",
                        type=str)
    parser.add_argument('--s2_loss_type', default="dsmvr",
                        type=str)
    parser.add_argument('--epochs', default=50000,
                        type=int)
    parser.add_argument('--lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--T', default=1.0,
                        type=float)
    parser.add_argument('--dt_steps', default=1000,
                        type=int)
    parser.add_argument('--x_samples', default=32,
                        type=int)
    parser.add_argument('--t_samples', default=128,
                        type=int)
    parser.add_argument('--repeats', default=8,
                        type=int)
    parser.add_argument('--t0_sample', default=0,
                        type=int)
    parser.add_argument('--t0', default=0.01,
                        type=float)
    parser.add_argument('--gamma', default=1.0,
                        type=float)
    parser.add_argument('--load_model', default=0,
                        type=int)
    parser.add_argument('--save_step', default=10,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% train for (x,y,t)

def train_score()->None:
    
    args = parse_args()

    T_sample_name = (args.t0_sample == 1)*"T"
    s1_path = f"scores/{args.manifold}{args.dim}/s1{T_sample_name}_{args.s1_loss_type}/"
    s2_path = f"scores/{args.manifold}{args.dim}/s2{T_sample_name}_{args.s2_loss_type}/"
    s1s2_path = f"scores/{args.manifold}{args.dim}/s1s2{T_sample_name}_{args.s2_loss_type}/"
    
    M, x0, sampling_method, generator_dim, layers, opt_val = load_manifold(args.manifold,
                                                                           args.dim)
    
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
    
    if not os.path.exists('scores/hpc/output/'):
        os.makedirs('scores/hpc/output/')
        
    if not os.path.exists('scores/hpc/error/'):
        os.makedirs('scores/hpc/error/')
        
    if args.t0_sample:
        t0 = args.t0
    else:
        t0 = 0.0
        
    data_generator = RiemannianBrownianGenerator(M=M,
                                                 x0 = x0,
                                                 dim = generator_dim,
                                                 Exp_map = lambda x, v: M.ExpEmbedded(x[0],v),
                                                 repeats = args.repeats,
                                                 x_samples = args.x_samples,
                                                 t_samples = args.t_samples,
                                                 T = args.T,
                                                 dt_steps = args.dt_steps,
                                                 t0 = t0,
                                                 method = sampling_method,
                                                 seed = args.seed
                                                 )

    if args.train_net == "s2":
        state = load_model(s1_path)
        rng_key = jran.PRNGKey(args.seed)
        s1 = lambda x,y,t: s1_model.apply(state.params,rng_key, jnp.hstack((x, y, t.reshape(-1,1))))
        
        if args.load_model:
            state_s2 = load_model(s2_path)
        else:
            state_s2 = None

        if not os.path.exists(s2_path):
            os.makedirs(s2_path)
        
        train_s2(M=M,
                 s1_model=s1,
                 s2_model=s2_model,
                 generator=data_generator,
                 state=state_s2,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 save_step=args.save_step,
                 save_path=s2_path,
                 seed=args.seed,
                 loss_type = args.s2_loss_type
                 )
    elif args.train_net == "s1s2":
        rng_key = jran.PRNGKey(2712)
        
        if not os.path.exists(s1_path):
            state_s1_params = load_model(s1_path).params
        else:
            state_s1_params=None
        if not os.path.exists(s1s2_path):
            os.makedirs(s1s2_path)
            
        if args.load_model:
            state_s1s2 = load_model(s1s2_path)
        else:
            state_s1s2 = None
        
        train_s1s2(M=M,
                 s1s2_model=s1s2_model,
                 generator=data_generator,
                 state=state_s1s2,
                 s1_params=state_s1_params,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 save_step=args.save_step,
                 gamma=args.gamma,
                 save_path=s1s2_path,
                 seed=args.seed,
                 loss_type = args.s2_loss_type
                 )
    else:
        if args.load_model:
            state = load_model(s1_path)
        else:
            state = None

        if not os.path.exists(s1_path):
            os.makedirs(s1_path)
            
        train_s1(M=M,
                 model=s1_model,
                 generator=data_generator,
                 state =state,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 save_step=args.save_step,
                 save_path=s1_path,
                 loss_type=args.s1_loss_type,
                 seed=args.seed
                 )
    
    
    return

#%% Main

if __name__ == '__main__':
        
    train_score()
    