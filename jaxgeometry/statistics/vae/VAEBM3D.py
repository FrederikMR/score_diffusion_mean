#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:05:11 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax,
from jaxgeometry.setup import *
from jaxgeometry.manifolds import Latent

#jaxgeometry
from jaxgeometry.integration import dts, dWs
from jaxgeometry.stochastics import product_sde, Brownian_coords

from jax.nn import elu, sigmoid, swish

#%% VAE Output

class VAEOutput(NamedTuple):
  z: Array
  x_hat: Array
  mean: Array
  t: Array

#%% Other

@dataclasses.dataclass
class ScoreNet(hk.Module):
    
    dim:int
    layers:list
    
    def model(self)->object:
        
        model = []
        for l in self.layers:
            model.append(hk.Linear(l))
            model.append(tanh)
            
        model.append(hk.Linear(self.dim))
        
        return hk.Sequential(model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        x_new = x.T
        x1 = x_new[:self.dim].T
        x2 = x_new[self.dim:(2*self.dim)].T
        t = x_new[-1]
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x_new[-1].reshape(shape)
            
        grad_euc = (x1-x2)/t
      
        return self.model()(x)+grad_euc

@dataclasses.dataclass
class Encoder(hk.Module):
        
    latent_dim : int = 2

    def __call__(self, x:Array) -> Tuple[Array, Array]:

        x = x.reshape(-1,3)
        z = swish(hk.Linear(output_size=100)(x))
        z = swish(hk.Linear(output_size=100)(z))
        
        mu = hk.Linear(output_size=self.latent_dim)(z)
        t = sigmoid(hk.Linear(output_size=1)(z))

        return mu, t

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
                 decoder:Decoder
                 ):
        super(VAEBM, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x: Array) -> VAEOutput:
      """Forward pass of the variational autoencoder."""
      x = x.astype(jnp.float32)
      mu, log_std = self.encoder(x)
      t = log_std.reshape(-1,1)
      
      std = t*jnp.ones((len(t), 1))
      z = mu+std*jrandom.normal(hk.next_rng_key(), mu.shape)
      x_hat = self.decoder(z)
    
      return VAEOutput(z, x_hat, mu, std, std)

class VAEBM(hk.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder
                 ):
        super(VAEBM, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        
        F = lambda z: decoder(z[0].reshape(-1,2))        
        M = Latent(dim=2, emb_dim=3, F=F, invF=None)
        Brownian_coords(M)
        
        self.M = M
        
    def Jf(self,z):
        
        return jacfwd(self.decoder(z.reshape(-1,2)))(z)
        
    def G(self,z):
        
        Jf = self.Jf(z)
        
        return jnp.dot(Jf.T,Jf)
    
    def DG(self,z):
        
        return jacfwd(self.G)(z)
    
    def Ginv(self,z):
        
        return jnp.linalg.inv(G(z))
    
    def Chris(self,z):
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def euclidean_sample(self, mu:Array, t:Array):
        
        return mu+t*jran.normal(hk.next_rng_key(), mu.shape)
    
    def taylor_sample(self, mu:Array, t:Array):
        
        def step(carry, step):
            
            t,z = carry
            dt, dW = step
            
            t += dt
            ginv = self.Ginv(z)
            
            stoch = jnp.dot(ginv, dW)
            
            z += stoch
            
            return ((t,z),)*2
        
        dt = vmap(lambda t: dts(t,100))(t).squeeze()
        N_data = mu.shape[0]
        dW = vmap(lambda dt: dWs(self.encoder.latent_dim,dt))(dt).reshape(-1,N_data,self.encoder.latent_dim)
        
        #vmap(lambda mu,dt,dW: lax.scan(step, init=(mu,0.0), xs=(dt,dW)))(mu,dt,jnp.transpose(dW, axis=(1,0,2)))
        val, _ =lax.scan(lambda carry, step: vmap(lambda t,z,dt,dW: step((t,z),(dt,dW)))(carry[0],carry[1],step[0],step[1]),
                         init=(mu,t), xs=(dt,dW)
                         )

        return val[1]
  
    def local_sample(self, mu:Array, t:Array)->Array:
        
        def step(carry, step):
            
            t,z = carry
            dt, dW = step
            
            t += dt
            ginv = self.Ginv(z)
            Chris = self.Chris(z)
            
            stoch = jnp.dot(ginv, dW)
            det = 0.5*jnp.einsum('jk,ijk->i', ginv, chris)
            
            z += det+stoch
            
            return ((t,z),)*2
        
        dt = vmap(lambda t: dts(t,100))(t).squeeze()
        N_data = mu.shape[0]
        dW = vmap(lambda dt: dWs(self.encoder.latent_dim,dt))(dt).reshape(-1,N_data,self.encoder.latent_dim)
        
        #vmap(lambda mu,dt,dW: lax.scan(step, init=(mu,0.0), xs=(dt,dW)))(mu,dt,jnp.transpose(dW, axis=(1,0,2)))
        val, _ =lax.scan(lambda carry, step: vmap(lambda t,z,dt,dW: step((t,z),(dt,dW)))(carry[0],carry[1],step[0],step[1]),
                         init=(mu,t), xs=(dt,dW)
                         )

        return val[1]

    def __call__(self, x: Array, sample_method='Local') -> VAEOutput:
        """Forward pass of the variational autoencoder."""
        x = x.astype(jnp.float32)
        mu, t = self.encoder(x)
        
        if sample_method == 'Local':
            z = self.local_sample(mu, t)
        elif sample_method == 'Taylor':
            z = self.taylor_sample(mu, t)
        elif sample_method == 'Euclidean':
            z = self.euclidean_sample(mu, t)
        else:
            raise ValueError("Invalid sampling method. Choose either: Local, Taylor, Euclidean")          

        x_hat = self.decoder(z)
          
        return VAEOutput(z, x_hat, mu, t)

#%% Transformed model
    
@hk.transform
def model(x):
    
    vae = VAEBM(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae(x)

#%% Transformed Encoder model
    
@hk.transform
def model_encoder(x):
    
    vae = VAEBM(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae.encoder(x)[0]

#%% Transformed Decoder model
    
@hk.transform
def model_decoder(z):
    
    vae = VAEBM(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae.decoder(z)

@hk.transform
def score_model(z):
    
    score = ScoreNet(
    dim=2,
    layers=[50,100,200,200,100,50],
    )
  
    return vae.decoder(z)
