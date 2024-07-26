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
from jaxgeometry.statistics.score_matching import train_s1, train_s2, train_s1s2, train_t, train_p, TMSampling, LocalSampling, \
    EmbeddedSampling, ProjectionSampling
from jaxgeometry.statistics.score_matching.model_loader import load_model
from ManLearn.train_MNIST import load_dataset as load_mnist

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="Sphere",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--s1_loss_type', default="dsm",
                        type=str)
    parser.add_argument('--s2_loss_type', default="dsm",
                        type=str)
    parser.add_argument('--load_model', default=0,
                        type=int)
    parser.add_argument('--T_sample', default=0,
                        type=int)
    parser.add_argument('--t0', default=0.1,
                        type=float)
    parser.add_argument('--gamma', default=1.0,
                        type=float)
    parser.add_argument('--train_net', default="s1",
                        type=str)
    parser.add_argument('--max_T', default=1.0,
                        type=float)
    parser.add_argument('--lr_rate', default=0.001,
                        type=float)
    parser.add_argument('--epochs', default=200000,
                        type=int)
    parser.add_argument('--warmup_epochs', default=1000,
                        type=int)
    parser.add_argument('--x_samples', default=1, #32
                        type=int)
    parser.add_argument('--t_samples', default=100,#128
                        type=int)
    parser.add_argument('--repeats', default=1024, #32
                        type=int)
    parser.add_argument('--dt_steps', default=100,
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

    T_sample_name = (args.T_sample == 1)*"T"
    st_path = f"scores/{args.manifold}{args.dim}/st/"
    s1_path = f"scores/{args.manifold}{args.dim}/s1{T_sample_name}_{args.s1_loss_type}/"
    s2_path = f"scores/{args.manifold}{args.dim}/s2{T_sample_name}_{args.s2_loss_type}/"
    s1s2_path = f"scores/{args.manifold}{args.dim}/s1s2{T_sample_name}_{args.s2_loss_type}/"
    
    M, x0, sampling_method, generator_dim, layers, opt_val = load_manifold(args.manifold,
                                                                           args.dim)
    layers_s1, layers_s2 = layers
    
    s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers_s1)(x))
    st_model = hk.transform(lambda x: models.MLP_t(dim=generator_dim, layers=layers_s1)(x))
    if "diag" in args.s2_loss_type:
        s2_model = hk.transform(lambda x: models.MLP_diags2(layers_alpha=layers_s2, layers_beta=layers_s2,
                                                        dim=generator_dim, 
                                                        r = max((generator_dim-1)//2,1))(x))
    
        @hk.transform
        def s1s2_model(x):
            
            s1s2 =  models.MLP_diags1s2(
                models.MLP_s1(dim=generator_dim, layers=layers_s1), 
                models.MLP_s2(layers_alpha=layers_s2, 
                              layers_beta=layers_s2,
                              dim=generator_dim,
                              r = max((generator_dim-1)//2,1))
                )
            
            return s1s2(x)
    else:    
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers_s2, layers_beta=layers_s1,
                                                        dim=generator_dim, 
                                                        r = max((generator_dim-1)//2,1))(x))
    
        @hk.transform
        def s1s2_model(x):
            
            s1s2 =  models.MLP_s1s2(
                models.MLP_s1(dim=generator_dim, layers=layers_s1), 
                models.MLP_s2(layers_alpha=layers_s2, 
                              layers_beta=layers_s2,
                              dim=generator_dim,
                              r = max((generator_dim-1)//2,1))
                )
             
            return s1s2(x)
        
    if args.train_net == "t":
        t_samples = args.dt_steps
    else:
        t_samples = args.t_samples
        
    if sampling_method == 'LocalSampling':
        data_generator = LocalSampling(M=M,
                                       x0=x0,
                                       repeats=args.repeats,
                                       x_samples=args.x_samples,
                                       t_samples=t_samples,
                                       max_T=args.max_T,
                                       dt_steps=args.dt_steps,
                                       T_sample=args.T_sample,
                                       t0=args.t0
                                       )
    elif sampling_method == "EmbeddedSampling":
        data_generator = EmbeddedSampling(M=M,
                                          x0=x0,
                                          repeats=args.repeats,
                                          x_samples=args.x_samples,
                                          t_samples=t_samples,
                                          max_T=args.max_T,
                                          dt_steps=args.dt_steps,
                                          T_sample=args.T_sample,
                                          t0=args.t0
                                          )
    elif sampling_method == "ProjectionSampling":
        data_generator = ProjectionSampling(M=M,
                                            x0=(x0[1],x0[0]),
                                            dim=generator_dim,
                                            repeats=args.repeats,
                                            x_samples=args.x_samples,
                                            t_samples=t_samples,
                                            max_T=args.max_T,
                                            dt_steps=args.dt_steps,
                                            T_sample=args.T_sample,
                                            t0=args.t0
                                            )
    elif sampling_method == "TMSampling":
        data_generator = TMSampling(M=M,
                                    x0=(x0[1],x0[0]),
                                    dim=generator_dim,
                                    Exp_map=lambda x, v: M.ExpEmbedded(x[0],v),
                                    repeats=args.repeats,
                                    x_samples=args.x_samples,
                                    t_samples=t_samples,
                                    max_T=args.max_T,
                                    dt_steps=args.dt_steps,
                                    T_sample=args.T_sample,
                                    t0=args.t0
                                    )

    if args.train_net == "s2":
        state = load_model(s1_path)
        rng_key = jran.PRNGKey(2712)
        s1 = lambda x,y,t: s1_model.apply(state.params,rng_key, jnp.hstack((x, y, t)))
        
        if args.load_model:
            state_s2 = load_model(s1_path)
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
                 warmup_epochs=args.warmup_epochs,
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
                 warmup_epochs=args.warmup_epochs,
                 save_step=args.save_step,
                 gamma=args.gamma,
                 save_path=s1s2_path,
                 seed=args.seed,
                 loss_type = args.s2_loss_type
                 )
    elif args.train_net == "t":
        if args.load_model:
            state = load_model(st_path)
        else:
            state = None

        if not os.path.exists(st_path):
            os.makedirs(st_path)

        train_t(M=M,
                model=st_model,
                generator=data_generator,
                state=state,
                lr_rate=args.lr_rate,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                save_step=args.save_step,
                save_path=st_path,
                seed=args.seed
                )
    elif args.train_net == "s1p":
        s1p_path = f"scores/{args.manifold}{args.dim}/s1p{T_sample_name}_{args.s1_loss_type}/"
        s1p_model = hk.transform(lambda x: models.MLP_p(dim=generator_dim, layers=layers_s1)(x))
        if args.load_model:
            state = load_model(s1p_path)
        else:
            state = None

        if not os.path.exists(s1p_path):
            os.makedirs(s1p_path)

        train_p(M=M,
                model=s1p_model,
                generator=data_generator,
                state =state,
                lr_rate=args.lr_rate,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                save_step=args.save_step,
                save_path=s1p_path,
                loss_type=args.s1_loss_type,
                seed=args.seed
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
                 warmup_epochs=args.warmup_epochs,
                 save_step=args.save_step,
                 save_path=s1_path,
                 loss_type=args.s1_loss_type,
                 seed=args.seed
                 )
    
    
    return

#%% Main

if __name__ == '__main__':
        
    train_score()
    
