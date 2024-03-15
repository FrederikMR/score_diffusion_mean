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

import tensorflow as tf

from jaxgeometry.statistics.vae.VAEBM3D import ScoreNet, Encoder, Decoder, VAEBM
from jaxgeometry.statistics.score_matching import train_vaebm

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
    
    train_vaebm(vae_model = vae_model,
                decoder_model = decoder_model,
                score_model = score_model,
                vae_datasets = vae_datasets,
                mu0 = 0.0*jnp.zeros(2),
                t0 = 1.0*jnp.ones(1),
                dim = 2,
                emb_dim = 3,
                vae_state=None,
                score_state=None,
                lr_rate = 0.0002,
                burnin_epochs=100,
                joint_epochs=100,
                repeats=1,
                save_step=100,
                vae_optimizer=None,
                score_optimizer=None,
                vae_save_path = "",
                score_save_path = "",
                score_type = 'dsmvr',
                seed = 2712
                )
                
    
    print(X.shape)
    
    return

#%% Main

if __name__ == '__main__':
        
    train()