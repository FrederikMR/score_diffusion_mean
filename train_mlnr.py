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
import jax.random as jrandom
from jax import vmap, jacfwd
from jax import Array

from scipy import ndimage

from gp.gp import RM_EG

import numpy as np

#haiku
import haiku as hk

#argparse
import argparse

#scores
from models import models, neural_regression

import pickle

from typing import NamedTuple

#os
import os

from load_manifold import load_manifold

#jaxgeometry
from jaxgeometry.manifolds import *
from jaxgeometry.statistics.score_matching import train_mlnr
from jaxgeometry.statistics.score_matching import ScoreEvaluation
from jaxgeometry.statistics.score_matching.model_loader import load_model
from ManLearn.train_MNIST import load_dataset as load_mnist

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="Sphere",
                        type=str)
    parser.add_argument('--dim', default=3,
                        type=int)
    parser.add_argument('--s1_loss_type', default="dsm",
                        type=str)
    parser.add_argument('--s2_loss_type', default="dsm",
                        type=str)
    parser.add_argument('--dt_type', default="s1",
                        type=str)
    parser.add_argument('--load_model', default=0,
                        type=int)
    parser.add_argument('--min_t', default=0.01,
                        type=float)
    parser.add_argument('--max_t', default=1.0,
                        type=float)
    parser.add_argument('--lr_rate', default=0.001,
                        type=float)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--epochs', default=50000,
                        type=int)
    parser.add_argument('--warmup_epochs', default=1000,
                        type=int)
    parser.add_argument('--save_step', default=100,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% train for (x,y,t)

def train()->None:
    
    args = parse_args()
    
    rng_key = jrandom.PRNGKey(2712)
    
    mlnr_path = f"mlnr/{args.manifold}{args.dim}/"
    
    s1_path = f"scores/{args.manifold}{args.dim}/s1_{args.s1_loss_type}/"
    st_path = f"scores/{args.manifold}{args.dim}/st/"
    s2_path = f"scores/{args.manifold}{args.dim}/s2_{args.s2_loss_type}/"
    s1s2_path = f"scores/{args.manifold}{args.dim}/s1s2_{args.s2_loss_type}/"
    
    M, x0, method, generator_dim, layers, opt_val = load_manifold(args.manifold,args.dim)
    layers_s1, layers_s2 = layers
    if method == "LocalSampling":
        method = "Local"
    else:
        method = "Embedded"
    
    s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers_s1)(x))
    st_model = hk.transform(lambda x: models.MLP_t(dim=generator_dim, layers=layers_s1)(x))
    if "diag" in args.s2_loss_type:
        s2_model = hk.transform(lambda x: models.MLP_diags2(layers_alpha=layers_s2, layers_beta=layers_s2,
                                                        dim=generator_dim, 
                                                        r = max((generator_dim-1)//2,1))(x))
    else:    
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers_s2, layers_beta=layers_s1,
                                                        dim=generator_dim, 
                                                        r = max((generator_dim-1)//2,1))(x))
        
    if args.dt_type == "s1s2":
        s1_state = load_model(s1s2_path)
        s2_state = load_model(s1s2_path)
        
        s1_fun = lambda x,y,t: s1_model.apply(s1_state.params, rng_key, jnp.hstack((x,y,t)))
        s2_fun = lambda x,y,t: lax.stop_gradient(s2_model.apply(s2_state.params, rng_key, jnp.hstack((x,y,t))))
        st_fun = None
        
    elif args.dt_type == "t":
        st_state = load_model(st_path)
        s1_state = load_model(s1_path)
        
        s1_fun = lambda x,y,t: s1_model.apply(s1_state.params, rng_key, jnp.hstack((x,y,t)))
        s2_fun = None
        st_fun = lambda x,y,t: lax.stop_gradient(st_model.apply(st_state.params, rng_key, jnp.hstack((x,y,t))))
    else:
        s1_state = load_model(s1_path)
        
        s1_fun = lambda x,y,t: s1_model.apply(s1_state.params, rng_key, jnp.hstack((x,y,t)))
        s2_fun = None
        st_fun = None
    
    
    if not os.path.exists(mlnr_path):
        os.makedirs(mlnr_path)
        
    ScoreEval = ScoreEvaluation(M,
                                s1_model=s1_fun,#s1_fun, 
                                s2_model=None,#s2_fun,#s2_model_test2, 
                                st_model=None,
                                method=method, 
                                )
        
    def grady_log(x,y,t):
        
        if x.ndim == 1:
            return s1_fun(x,y,t)
        else:
            return vmap(lambda x1,y1,t1: s1_fun(x1,y1,t1))(x,y,t)
    
    def gradt_log(x,y,t):
        
        if x.ndim == 1:
            return ScoreEval.gradt_log(x,y,t)
        else:
            return vmap(lambda x1,y1,t1: ScoreEval.gradt_log(x1,y1,t1))(x,y,t)
        
        
    if args.manifold == "Euclidean":
        @hk.transform
        def mlnr_model(x):
            
            mlnr =  neural_regression.MLP_mlnr_R2(
                neural_regression.MLP_f_R2(dim=generator_dim, layers=layers_s1), 
                neural_regression.MLP_sigma(layers=layers_s1)
                )
             
            return mlnr(x)
        
        
        key, subkey = jrandom.split(rng_key)
        input_data = 1.0*jrandom.normal(key, shape=(50000,))
        key, subkey = jrandom.split(rng_key)
        eps = 0.1*jrandom.normal(key, shape=(50000,2))
        output_data = eps+jnp.stack((input_data**2, input_data**3)).T
        input_data = input_data.reshape(-1,1)
        np.savetxt(''.join((mlnr_path, 'input.csv')), input_data, delimiter=",")
        np.savetxt(''.join((mlnr_path, 'output.csv')), output_data, delimiter=",")
    elif args.manifold == "Sphere":
        
        @hk.transform
        def mlnr_model(x):
            
            mlnr =  neural_regression.MLP_mlnr_S2(
                neural_regression.MLP_f_S2(dim=generator_dim, layers=layers_s1), 
                neural_regression.MLP_sigma(layers=layers_s1)
                )
             
            return mlnr(x)
        
        data_path = '../data/AFLW2000/head_pose.pkl'
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        print(data_dict.keys())
            
        input_data = jnp.array(data_dict['train_features'])
        output_data = jnp.array(data_dict['train_labels'])
        
    else:
        raise ValueError("Datasets only defined for R2 (Euclidean with dim=2) and S3 (Sphere with dim=3)")
        
    train_mlnr(input_data=input_data,
               output_data=output_data,
               M=M,
               model=mlnr_model,
               grady_log=grady_log,
               gradt_log=gradt_log,
               state = None,
               lr_rate = args.lr_rate,
               batch_size=args.batch_size,
               epochs=args.epochs,
               warmup_epochs=args.warmup_epochs,
               save_step=args.save_step,
               optimizer=None,
               save_path = mlnr_path,
               min_t=args.min_t,
               max_t=args.max_t,
               seed=args.seed
               )
    
    
    return

#%% Main

if __name__ == '__main__':
        
    train()
    
