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

#%% VAE Loss Fun

#@partial(jit, static_argnames=['state_val', 'vae_apply_fn', 'training_type'])
def vae_euclidean_loss(params:hk.Params, state_val:dict, rng_key:Array, x, vae_apply_fn,
                       training_type="All")->Array:
    
    @jit
    def gaussian_likelihood(x, mu_xz, log_sigma_xz):
        
        dim = mu_xz.shape[-1]
        diff = x-mu_xz
        sigma_xz = jnp.exp(log_sigma_xz)
        mu_term = jnp.sum(jnp.einsum('ij,ij->ij', diff**2, 1/(sigma_xz**2)), axis=-1)
        var_term = 2*dim*jnp.sum(log_sigma_xz, axis=-1)
        
        loss = -0.5*(mu_term+var_term)
        
        return jnp.mean(loss)
    
    @jit
    def kl_divergence(z, mu_zx, log_t_zx, mu_z, log_t_z):
        
        dim = mu_zx.shape[-1]
        diff = z-mu_zx
        log_t_zx = log_t_zx.squeeze()
        log_t_z = log_t_z.squeeze()
        t_zx = jnp.exp(2*log_t_zx)
        t_z = jnp.exp(2*log_t_z)
        dist = jnp.sum(jnp.einsum('ij,i->ij', diff**2, 1/(t_zx)), axis=-1)
        log_qzx = -0.5*(dim*log_t_zx+dist)
        
        diff = z-mu_z
        dist = jnp.sum(jnp.einsum('ij,i->ij', diff**2, 1/(t_z)), axis=-1)
        log_pz = -0.5*(dim*log_t_z+dist)
        
        return jnp.mean(log_qzx-log_pz)

    z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, *_ = vae_apply_fn(params, x, rng_key, state_val)

    if training_type == "Encoder":
        sigma_xz = lax.stop_gradient(log_sigma_xz)
        mu_z = lax.stop_gradient(mu_z)
        t_z = lax.stop_gradient(log_t_z)
    elif training_type == "Decoder":
        mu_xz = lax.stop_gradient(mu_xz)
        t_zx = lax.stop_gradient(log_t_zx)
        mu_zx = lax.stop_gradient(mu_zx)

    rec_loss = gaussian_likelihood(x, mu_xz, log_sigma_xz)
    kld = kl_divergence(z, mu_zx, log_t_zx, mu_z, log_t_z)
    elbo = kld-rec_loss
    
    return elbo, (rec_loss, kld)
#%% VAE Riemannian Fun

#@partial(jit, static_argnames=['vae_state_val', 'vae_apply_fn', 'score_apply_fn', 'score_state', 'training_type'])
def vae_riemannian_loss(params:hk.Params, vae_state_val:dict, rng_key:Array, x:Array, vae_apply_fn,
                        score_apply_fn, score_state, training_type="All"):
    
    @jit
    def gaussian_likelihood(z:Array, mu_xz:Array, log_sigma_xz:Array):
        
        dim = mu_xz.shape[-1]
        diff = x-mu_xz
        sigma_xz = jnp.exp(log_sigma_xz)
        mu_term = jnp.sum(jnp.einsum('ij,ij->ij', diff**2, 1/(sigma_xz**2)), axis=-1)
        var_term = 2*dim*jnp.sum(log_sigma_xz, axis=-1)
        
        loss = -0.5*(mu_term+var_term)
        
        return loss
    
    @jit
    def kl_divergence(z:Array, s_logqzx:Array, s_logpz:Array):

        return jnp.einsum('...i,...i->...', s_logqzx, z)-jnp.einsum('...i,...i->...', s_logpz, z)
    
    @jit
    def loss_fun(z:Array, mu_xz:Array, log_sigma_xz:Array, s_logqzx:Array, s_logpz:Array):
        
        rec = gaussian_likelihood(z, mu_xz, log_sigma_xz)
        kld = kl_divergence(z, s_logqzx, s_logpz)
        
        return jnp.mean(rec-kld)
            
    z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, *_ = vae_apply_fn(params, x, rng_key, vae_state_val)
    
    t_zx = jnp.exp(2*log_t_zx)
    t_z = jnp.exp(2*log_t_z)
    
    s_logqzx = lax.stop_gradient(score_apply_fn(score_state.params, 
                                                jnp.hstack((mu_zx,z,t_zx)), 
                                                score_state.rng_key, 
                                                score_state.state_val))
    s_logpz = lax.stop_gradient(score_apply_fn(score_state.params, 
                                               jnp.hstack((mu_z, z, t_z)), 
                                               score_state.rng_key, 
                                               score_state.state_val))
    
    if training_type == "Encoder":
        sigma_xz = lax.stop_gradient(sigma_xz)
        mu_z = lax.stop_gradient(mu_z)
        t_z = lax.stop_gradient(t_z)
    elif training_type == "Decoder":
        mu_xz = lax.stop_gradient(mu_xz)
        t_zx = lax.stop_gradient(t_zx)
        mu_zx = lax.stop_gradient(mu_zx)
    loss = loss_fun(z, mu_xz, log_sigma_xz, s_logqzx, s_logpz)

    return loss

#%% Denoising Score Matching First Order

def dsm(s1_model:Callable[[Array, Array, Array], Array],
        x0:Array,
        xt:Array,
        t:Array,
        dW:Array,
        dt:Array,
        )->float:
    
    def f(x0,xt,t,dW,dt):
        
        s1 = s1_model(x0, xt, t)

        loss = dW/dt+s1
        
        return jnp.sum(loss*loss)
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Variance Reduction Denoising Score Matching First Order

def dsmvr(s1_model:object, 
          x0:Array,
          xt:Array,
          t:Array,
          dW:Array,
          dt:Array,
          )->float:
    
    def f(x0,xt,t,dW,dt):
        
        s1 = s1_model(x0,xt,t)
        s1p = s1_model(x0,x0,t)
        
        l1_loss = dW/dt+s1
        l1_loss = 0.5*jnp.dot(l1_loss,l1_loss)
        var_loss = jnp.dot(s1p,dW)/dt+jnp.dot(dW,dW)/(dt**2)
        
        return l1_loss-var_loss
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))