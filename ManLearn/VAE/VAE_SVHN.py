#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:14:54 2023

@author: fmry
"""

#%% Sources

#https://github.com/deepmind/dm-haiku/blob/main/examples/vae.py

#%% Modules

from ManLearn.initialize import *
  
#%% VAE Output

class VAEOutput(NamedTuple):
  z: Array
  x_hat: Array
  mean: Array
  std: Array
  
#%% Encoder

@dataclasses.dataclass
class Encoder(hk.Module):
        
    latent_dim : int = 32

    def __call__(self, x:Array) -> Tuple[Array, Array]:
        
        z = hk.Conv2D(output_channels = 32, kernel_shape=4, stride=2,
                                with_bias = False)(x)
        #z = hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset = True)(z, True)
        z = swish(z)
        
        z = hk.Conv2D(output_channels = 32, kernel_shape = 4, stride = 2,
                                with_bias = False)(z)
        #z = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(z, True)
        z = swish(z)
        
        z = hk.Conv2D(output_channels = 64, kernel_shape = 4, stride = 2,
                                with_bias = False)(z)
        #z = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(z, True)
        z = swish(z)
        
        z = hk.Conv2D(output_channels = 64, kernel_shape = 4, stride = 2,
                                with_bias = False)(z)
        #z = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(z, True)
        z = swish(z)
        
        z = hk.Linear(output_size = 256)(hk.Flatten()(z))
        #z = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(z, True)
        z = swish(z)
        
        mu = hk.Linear(output_size=self.latent_dim)(z)
        std = sigmoid(hk.Linear(output_size=self.latent_dim)(z))

        return mu, std

#%% Decoder

@dataclasses.dataclass
class Decoder(hk.Module):

    def __call__(self, z:Array) -> Array:
        
        x_hat = hk.Linear(output_size=256)(z)
        #x_hat = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(x_hat, True)
        x_hat = swish(x_hat.reshape(-1, 1, 1, 256))
        
        x_hat = hk.Conv2DTranspose(output_channels = 64, kernel_shape = 4, stride = 2)(x_hat)
        #x_hat = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(x_hat, True)
        x_hat = swish(x_hat)

        x_hat = hk.Conv2DTranspose(output_channels = 64, kernel_shape = 4, stride = 2)(x_hat)
        #x_hat = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(x_hat, True)
        x_hat = swish(x_hat)

        x_hat = hk.Conv2DTranspose(output_channels = 32, kernel_shape = 4, stride = 2)(x_hat)
        #x_hat = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(x_hat, True)
        x_hat = swish(x_hat)
        
        x_hat = hk.Conv2DTranspose(output_channels = 32, kernel_shape = 4, stride = 2)(x_hat)
        #x_hat = hk.BatchNorm(decay_rate = 0.9, create_scale = True, create_offset = True)(x_hat, True)
        x_hat = swish(x_hat)
        
        x_hat = hk.Conv2DTranspose(output_channels = 3, kernel_shape = 4, stride = 2)(x_hat)
        
        return x_hat

#%% VAE

@dataclasses.dataclass
class VariationalAutoEncoder(hk.Module):
    
    encoder: Encoder
    decoder: Decoder
        
    def __call__(self, x:Array)->VAEOutput:
        
        x = x.astype(jnp.float32)
        
        mu, std = self.encoder(x)
                
        z = mu+std*jran.normal(hk.next_rng_key(), mu.shape)
        x_hat = self.decoder(z)
    
        return VAEOutput(z, x_hat, mu, std)
#%% Transformed model
    
@hk.transform
def model(x):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=32),
    decoder=Decoder(),
    )
  
    return vae(x)

#%% Transformed Encoder model
    
@hk.transform
def model_encoder(x):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=32),
    decoder=Decoder(),
    )
  
    return vae.encoder(x)[0]

#%% Transformed Decoder model
    
@hk.transform
def model_decoder(z):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=32),
    decoder=Decoder(),
    )
  
    return vae.decoder(z)
    


