#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:56:46 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
import jax.random as jrandom

import haiku as hk

import os

import tensorflow as tf

from jaxgeometry.statistics.vae.VAEBM3D import ScoreNet, Encoder, Decoder, VAEBM
from jaxgeometry.statistics.score_matching import train_vaebm, pretrain_vae, pretrain_scores

#%% Code

def train():
        
    @hk.transform
    def vae_model(x):
        
        
        vae = VAEBM(
        encoder=Encoder(latent_dim=2),
        decoder=Decoder(),
        )
      
        return vae(x)

    @hk.transform
    def score_model(x):
        
        score = ScoreNet(
        dim=2,
        layers=[50,100,200,200,100,50],
        )
      
        return score(x)
    
    @hk.transform
    def decoder_model(z):
        
        decoder = Decoder()
      
        return decoder(z)
    
    N_data = 50000
    rng_key = jrandom.key(2712)
    
    rng_key, subkey = jrandom.split(rng_key)
    theta = jrandom.uniform(subkey, shape=(N_data,), minval=0.0, maxval=2*jnp.pi)
    rng_key, subkey = jrandom.split(rng_key)
    std = 0.01
    eps = std*jrandom.normal(subkey, shape=(N_data,))
    
    x1, x2, x3 = jnp.cos(theta), jnp.sin(theta), eps
    
    X = jnp.stack((x1,x2,x3)).T
    
    vae_datasets = tf.data.Dataset.from_tensor_slices(X).shuffle(buffer_size=10 * 100, seed=2712)\
        .batch(100).prefetch(buffer_size=5).repeat().as_numpy_iterator()
        
    vae_save_path = 'vaebm/vae/'
    score_save_path = 'vaebm/score/'
    if not os.path.exists(vae_save_path):
        os.makedirs(vae_save_path)
    if not os.path.exists(score_save_path):
        os.makedirs(score_save_path)
        
    if not os.path.exists('scores/output/'):
        os.makedirs('scores/output/')
        
    pretrain_vae(vae_model=vae_model,
                     data_generator=vae_datasets,
                     lr_rate = 0.002,
                     save_path = vae_save_path,
                     split = 1/3,
                     vae_state = None,
                     epochs=1000,
                     save_step = 100,
                     vae_optimizer = None,
                     seed=2712,
                     )
    
    print(X.shape)
    
    return

#%% Main

if __name__ == '__main__':
        
    train()