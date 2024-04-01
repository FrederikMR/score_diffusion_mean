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
from jaxgeometry.stochastics import product_sde, Brownian_coords

from jax.nn import elu, sigmoid, swish, tanh

#%% VAE Output

class VAEOutput(NamedTuple):
  z: Array
  mu_xz: Array
  sigma_xz: Array
  mu_zx: Array
  t_zx: Array
  mu_z: Array
  t_z: Array

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
    
    def mu_layer(self, z:Array)->Array:
        
        return hk.Linear(output_size=self.latent_dim)(z)
    
    def t_layer(self, z:Array)->Array:
        
        return sigmoid(hk.Linear(output_size=1)(z))

    def __call__(self, x:Array) -> Tuple[Array, Array]:

        x = x.reshape(-1,3)
        z = swish(hk.Linear(output_size=100)(x))
        z = swish(hk.Linear(output_size=100)(z))
        
        mu_zx = self.mu_layer(z)
        t_zx = self.t_layer(z)

        return mu_zx, t_zx

@dataclasses.dataclass
class Decoder(hk.Module):
  """Decoder model."""

  def __call__(self, z: Array) -> Array:

        x_hat = swish(hk.Linear(output_size=100)(z))
        mu_xz = swish(hk.Linear(output_size=3)(x_hat))
        sigma_xz = swish(hk.Linear(output_size=3)(x_hat))
        
        return mu_xz, sigma_xz

class VAEBM(hk.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 seed:int=2712
                 ):
        super(VAEBM, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.key = jrandom.key(seed)
        
    def muz(self, z:Array)->Array:
        
        z = swish(hk.Linear(output_size=100)(z))
        mu_z = hk.Linear(output_size=2)(z)
        
        return mu_z
    
    def tz(self, z:Array)->Array:
        
        z = swish(hk.Linear(output_size=100)(z))
        t_z = hk.Linear(output_size=1)(z)
        
        return t_z
        
    def dts(self, T:float=1.0,n_steps:int=n_steps)->Array:
        """time increments, deterministic"""
        return jnp.array([T/n_steps]*n_steps)

    def dWs(self,d:int,_dts:Array=None,num:int=1)->Array:
        """
        standard noise realisations
        time increments, stochastic
        """
        keys = jrandom.split(self.key,num=num+1)
        self.key = keys[0]
        subkeys = keys[1:]
        if _dts == None:
            _dts = self.dts()
        if num == 1:
            return jnp.sqrt(_dts)[:,None]*jrandom.normal(subkeys[0],(_dts.shape[0],d))
        else:
            return vmap(lambda subkey: jnp.sqrt(_dts)[:,None]*jrandom.normal(subkey,(_dts.shape[0],d)))(subkeys) 
        
    def Jf(self,z):
        
        return jacfwd(lambda z: self.decoder(z.reshape(-1,2)).reshape(-1))(z)
        
    def G(self,z):
        
        Jf = self.Jf(z)
        
        return jnp.dot(Jf.T,Jf)
    
    def DG(self,z):
        
        return jacfwd(self.G)(z)
    
    def Ginv(self,z):
        
        return jnp.linalg.inv(self.G(z))
    
    def Chris(self,z):
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def normal_sample(self, mu:Array, t:Array):
        
        return mu+t*jrandom.normal(hk.next_rng_key(), mu.shape)
    
    def taylor_sample(self, mu:Array, t:Array):
        
        def sample(carry, step):
            
            t,z = carry
            dt, dW = step
            
            t += dt
            ginv = self.Ginv(z)
            
            stoch = jnp.dot(ginv, dW)
            
            z += stoch
            
            return ((t,z),)*2
        
        dt = hk.vmap(lambda t: self.dts(t,100), split_rng=False)(t).squeeze()
        N_data = mu.shape[0]
        dW = hk.vmap(lambda dt: self.dWs(self.encoder.latent_dim,dt),
                     split_rng=False)(dt).reshape(-1,N_data,self.encoder.latent_dim)
        
        #vmap(lambda mu,dt,dW: lax.scan(step, init=(mu,0.0), xs=(dt,dW)))(mu,dt,jnp.transpose(dW, axis=(1,0,2)))
        val, _ =hk.scan(lambda carry, step: hk.vmap(lambda t,z,dt,dW: sample((t,z),(dt,dW)),
                                                        split_rng=False)(carry[0],carry[1],step[0],step[1]),
                         init=(jnp.zeros_like(t),mu), xs=(dt,dW)
                         )

        return val[1]
    
    def local_sample(self, mu:Array, t:Array)->Array:
        
        def sample(carry, step):
            
            t,z = carry
            dt, dW = step
            
            t += dt
            ginv = self.Ginv(z)
            Chris = self.Chris(z)
            
            stoch = jnp.dot(ginv, dW)
            det = 0.5*jnp.einsum('jk,ijk->i', ginv, Chris)
            
            z += det+stoch
            
            t = t.astype(jnp.float32)
            z = z.astype(jnp.float32)
            
            return ((t,z),)*2
        
        dt = hk.vmap(lambda t: self.dts(t,100), split_rng=False)(t).squeeze()
        N_data = mu.shape[0]
        dW = hk.vmap(lambda dt: self.dWs(self.encoder.latent_dim,dt),
                     split_rng=False)(dt).reshape(-1,N_data,self.encoder.latent_dim)

        val, _ =hk.scan(lambda carry, step: hk.vmap(lambda t,z,dt,dW: sample((t,z),(dt,dW)), 
                                                        split_rng=False)(carry[0],carry[1],step[0],step[1]),
                         init=(jnp.zeros_like(t),mu), xs=(dt,dW)
                         )

        return val[1]

    def __call__(self, x: Array, sample_method='Local') -> VAEOutput:
        """Forward pass of the variational autoencoder."""
        x = x.astype(jnp.float32)
        mu_zx, t_zx = self.encoder(x)
        
        if sample_method == 'Local':
            z = self.local_sample(mu, t)
        elif sample_method == 'Taylor':
            z = self.taylor_sample(mu, t)
        elif sample_method == 'Euclidean':
            z = self.euclidean_sample(mu, t)
        else:
            raise ValueError("Invalid sampling method. Choose either: Local, Taylor, Euclidean")
            
        mu_z, t_z = self.muz(z), self.tz(z)

        mu_xz, sigma_xz = self.decoder(z)

        return VAEOutput(z, mu_xz, sigma_xz, mu_zx, t_zx, mu_z, t_z)

#%% Transformed model
    
@hk.transform
def vae_model(x):
    
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
def score_model(x):
    
    score = ScoreNet(
    dim=2,
    layers=[50,100,200,200,100,50],
    )
  
    return score(x)
