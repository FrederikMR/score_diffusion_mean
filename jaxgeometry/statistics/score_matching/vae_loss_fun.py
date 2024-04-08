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
        t_zx = jnp.exp(log_t_zx)
        t_z = jnp.exp(log_t_z)
        dist = jnp.sum(jnp.einsum('ij,i->ij', diff**2, 1/(t_zx**2)), axis=-1)
        log_qzx = -0.5*(dim*log_t_zx+dist)
        
        diff = z-mu_z
        dist = jnp.sum(jnp.einsum('ij,i->ij', diff**2, 1/(t_z**2)), axis=-1)
        log_pz = -0.5*(dim*log_t_z+dist)
        
        return jnp.mean(log_qzx-log_pz)

    z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z = vae_apply_fn(params, x, rng_key, state_val)

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
    def gaussian_likelihood(params:hk.Params, z:Array, mu_xz:Array, log_sigma_xz:Array):
        
        dim = mu_xz.shape[-1]
        diff = x-mu_xz
        sigma_xz = jnp.exp(log_sigma_xz)
        mu_term = jnp.sum(jnp.einsum('ij,ij->ij', diff**2, 1/(sigma_xz**2)), axis=-1)
        var_term = 2*dim*jnp.sum(log_sigma_xz, axis=-1)
        
        loss = -0.5*(mu_term+var_term)
        
        return jnp.mean(loss)
    
    @jit
    def kl_divergence(params:hk.Params, s_logqzx:Array, s_logpz:Array):
        
        z, *_ = vae_apply_fn(params, x, rng_key, vae_state_val)
        
        return jnp.mean(jnp.einsum('...i,...i->...', s_logqzx, z)-jnp.einsum('...i,...i->...', s_logpz, z))
            
    z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z = vae_apply_fn(params, x, rng_key, vae_state_val)
    
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
        
    kl_grad = grad(kl_divergence)(params, s_logqzx, s_logpz)
    rec_grad = grad(gaussian_likelihood)(params, z, mu_xz, log_sigma_xz)
    
    gradient = {layer: {name: kl_grad[layer][name]-rec_grad[layer][name] for name,w in val.items()} \
                    for layer,val in kl_grad.items()}

    return gradient