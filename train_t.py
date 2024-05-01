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
from jaxgeometry.statistics.score_matching.traint import train_t
from jaxgeometry.statistics.score_matching.generators_t import TMSampling, LocalSampling, EmbeddedSampling, \
    ProjectionSampling
from jaxgeometry.statistics.score_matching.model_loader import load_model
from ManLearn.train_MNIST import load_dataset as load_mnist

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="HypParaboloid",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--load_model', default=0,
                        type=int)
    parser.add_argument('--max_T', default=1.0,
                        type=float)
    parser.add_argument('--lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--epochs', default=50000,
                        type=int)
    parser.add_argument('--x_samples', default=32,
                        type=int)
    parser.add_argument('--repeats', default=8,
                        type=int)
    parser.add_argument('--samples_per_batch', default=16,
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
    
    N_sim = args.x_samples*args.repeats
    st_path = f"scores/{args.manifold}{args.dim}/st/"

    M, x0, sampling_method, generator_dim, layers, opt_val = load_manifold(args.manifold,
                                                                           args.dim)
    
    s1_model = hk.transform(lambda x: models.MLP_t(dim=generator_dim, layers=layers)(x))

    batch_size = args.x_samples*args.dt_steps*args.repeats
        
    if sampling_method == 'LocalSampling':
        data_generator = LocalSampling(M=M,
                                       x0=x0,
                                       repeats=args.repeats,
                                       x_samples=args.x_samples,
                                       N_sim=N_sim,
                                       max_T=args.max_T,
                                       dt_steps=args.dt_steps,
                                       )
    elif sampling_method == "EmbeddedSampling":
        data_generator = EmbeddedSampling(M=M,
                                          x0=x0,
                                          repeats=args.repeats,
                                          x_samples=args.x_samples,
                                          t_samples=args.t_samples,
                                          N_sim=N_sim,
                                          max_T=args.max_T,
                                          dt_steps=args.dt_steps,
                                          T_sample=args.T_sample,
                                          t=args.t
                                          )
    elif sampling_method == "ProjectionSampling":
        data_generator = ProjectionSampling(M=M,
                                            x0=(x0[1],x0[0]),
                                            dim=generator_dim,
                                            repeats=args.repeats,
                                            x_samples=args.x_samples,
                                            t_samples=args.t_samples,
                                            N_sim=N_sim,
                                            max_T=args.max_T,
                                            dt_steps=args.dt_steps,
                                            T_sample=args.T_sample,
                                            t=args.t
                                            )
    elif sampling_method == "TMSampling":
        data_generator = TMSampling(M=M,
                                    x0=(x0[1],x0[0]),
                                    dim=generator_dim,
                                    Exp_map=lambda x, v: M.ExpEmbedded(x[0],v),
                                    repeats=args.repeats,
                                    x_samples=args.x_samples,
                                    N_sim=N_sim,
                                    max_T=args.max_T,
                                    dt_steps=args.dt_steps,
                                    )

    if args.load_model:
        state = load_model(st_path)
    else:
        state = None

    if not os.path.exists(st_path):
        os.makedirs(st_path)

    train_t(M=M,
            model=s1_model,
            generator=data_generator,
            N_dim=generator_dim,
            dW_dim=generator_dim,
            batch_size=batch_size,
            state =state,
            lr_rate=args.lr_rate,
            epochs=args.epochs,
            save_step=args.save_step,
            save_path=st_path,
            seed=args.seed
            )
    
    
    return

#%% Main

if __name__ == '__main__':
        
    train_score()
    
