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

        x = x.reshape(-1,3)
        z = swish(hk.Linear(output_size=100)(x))
        z = swish(hk.Linear(output_size=100)(z))
        
        mu = hk.Linear(output_size=self.latent_dim)(z)
        log_std = hk.Linear(output_size=1)(z)

        return mu, log_std

@dataclasses.dataclass
class Decoder(hk.Module):
  """Decoder model."""

  def __call__(self, z: Array) -> Array:

        x_hat = swish(hk.Linear(output_size=100)(z))
        x_hat = swish(hk.Linear(output_size=3)(x_hat))
        
        return x_hat

class VAE(hk.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 score:ScoreNet):

        self.encoder = encoder
        self.decoder = decoder
        self.score = score
        
        M = Latent(self.decoder(z))
        Brownian_coords(M)
        (product, sde_product, chart_update_product) = product_sde(M,
                                                                   M.sde_Brownian_coords,
                                                                   M.chart_update_Brownian_coords)
        self.M = M
        self.product = product
  
    def sample(self, mu:Array, t:Array)->Array:
        
        (ts,xss,chartss,*_) = self.product((mu, jnp.zeros(len(mu))),self._dts,dW,jnp.repeat(1.,self.N_sim))
        
        return xss

    def __call__(self, x: Array) -> VAEOutput:
      """Forward pass of the variational autoencoder."""
      x = x.astype(jnp.float32)
      mu, log_std = self.encoder(x)
      
      t = jnp.exp(log_std)
      samples = self.sample(mu, t)
      s1 = score(jnp.hstack((mu, samples, t)))
      z = samples[0]
      x_hat = self.decoder(z)
      
      M = Latent(self.decoder(z))
      Brownian_coords(M)
      (product, sde_product, chart_update_product) = product_sde(M,
                                                                 M.sde_Brownian_coords,
                                                                 M.chart_update_Brownian_coords)
      self.M = M
      self.product = product
    
      return VAEOutput(z, x_hat, mu, std)

#%% Transformed model
    
@hk.transform
def model(x):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    score=ScoreNet()
    )
  
    return vae(x)

#%% Transformed Encoder model
    
@hk.transform
def model_encoder(x):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    score=ScoreNet()
    )
  
    return vae.encoder(x)[0]

#%% Transformed Decoder model
    
@hk.transform
def model_decoder(z):
    
    vae = VariationalAutoEncoder(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    score=ScoreNet()
    )
  
    return vae.decoder(z)
