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

import numpy as np

#argparse
import argparse

import os

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data', default="Circle3D",
                        type=str)
    parser.add_argument('--std', default=0.01,
                        type=float)
    parser.add_argument('--N_data', default=50000,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% Load Data

def generate_data():
    
    args = parse_args()
    
    data_path = 'data/vae/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if args.data == "Circle3D":
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        data_path = ''.join((data_path, f'{args.data}'))
        rng_key = jrandom.key(args.seed)
        
        rng_key, subkey = jrandom.split(rng_key)
        theta = jrandom.uniform(subkey, shape=(args.N_data,), minval=0.0, maxval=2*jnp.pi)
        rng_key, subkey = jrandom.split(rng_key)
        eps = args.std*jrandom.normal(subkey, shape=(3,args.N_data,))
        
        x1, x2, x3 = jnp.cos(theta)+1.0+eps[0], jnp.sin(theta)+1.0+eps[1], eps[2]+1.0
        
        X = jnp.stack((x1,x2,x3)).T
        
        np.save(data_path, X)
    else:
        print("NOT Implemented data type")
    
    return

#%% Main

if __name__ == '__main__':
    
    generate_data()