#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:05:11 2023

@author: fmry
"""

#%% Sources

#%% Modules

from ManLearn.initialize import *

#%% VAE Output

class VAEOutput(NamedTuple):
  z: Array
  x_hat: Array
  mean: Array
  std: Array

#%% Other

@dataclasses.dataclass
class Encoder(hk.Module):
        
    latent_dim : int = 2

    def __call__(self, x:Array) -> Tuple[Array, Array]:

        x = x.reshape(-1,28,28,1)
        z = hk.Conv2D(output_channels = 64, kernel_shape=4, stride=2,
                                with_bias = False)(x)
        z = swish(z)
        
        z = hk.Conv2D(output_channels = 64, kernel_shape = 4, stride=2, 
                                with_bias = False)(z)
        z = swish(z)
        
        z = hk.Conv2D(output_channels = 64, kernel_shape = 4, stride=1, 
                                with_bias = False)(z)
        z = swish(z)
        
        z = z.reshape(z.shape[0], -1)
        z = hk.Linear(output_size = 256)(hk.Flatten()(z))
        z = swish(z)
        
        mu = hk.Linear(output_size=self.latent_dim)(z)
        std = sigmoid(hk.Linear(output_size=self.latent_dim)(z))

        return mu, std

@dataclasses.dataclass
class Decoder(hk.Module):
  """Decoder model."""

  def __call__(self, z: Array) -> Array:

        x_hat = swish(hk.Linear(output_size=256)(z)).reshape(-1,1,1,256)
        x_hat = swish(hk.Linear(output_size=49)(x_hat)).reshape(-1,7,7,1)
        
        x_hat = hk.Conv2DTranspose(output_channels = 64, kernel_shape = 4, stride = 2)(x_hat)
        x_hat = swish(x_hat)
        
        x_hat = hk.Conv2DTranspose(output_channels = 64, kernel_shape = 4, stride = 1)(x_hat)
        x_hat = swish(x_hat)
        
        x_hat = hk.Conv2DTranspose(output_channels = 32, kernel_shape = 4, stride = 1)(x_hat)
        x_hat = swish(x_hat).reshape(x_hat.shape[0], -1)
        
        x_hat = hk.Linear(output_size=28*28)(x_hat)
        
        return x_hat.reshape(-1,28,28,1)

@dataclasses.dataclass
class VariationalAutoEncoder(hk.Module):
  """Main VAE model class."""

  encoder: Encoder
  decoder: Decoder

  def __call__(self, x: Array) -> VAEOutput:
    """Forward pass of the variational autoencoder."""
    x = x.astype(jnp.float32)
    mu, std = self.encoder(x)
    z = mu + std * jran.normal(hk.next_rng_key(), mu.shape)
    x_hat = self.decoder(z)

    return VAEOutput(z, x_hat, mu, std)

#%% Transformed model
    
@hk.transform
def model(x):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae(x)

#%% Transformed Encoder model
    
@hk.transform
def model_encoder(x):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae.encoder(x)[0]

#%% Transformed Decoder model
    
@hk.transform
def model_decoder(z):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae.decoder(z)
