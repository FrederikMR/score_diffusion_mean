#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources


#%% Modules

from jaxgeometry.setup import *
from .model_loader import save_model
from .loss_fun import *
from .generators import LocalSampling
from jaxgeometry.manifolds import Latent

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    rng_key: Array

#%% Score Training
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_vaebm(vae_model:object,
                decoder_model:object,
                score_model:object,
                vae_datasets:object,
                mu0:Array,
                t0:Array,
                dim:int,
                emb_dim:int,
                batch_size:int,
                vae_state:TrainingState = None,
                score_state:TrainingState = None,
                lr_rate:float = 0.001,
                burnin_epochs:int=100,
                joint_epochs:int=100,
                repeats:int=1,
                save_step:int=100,
                vae_optimizer:object=None,
                score_optimizer:object=None,
                vae_save_path:str = "",
                score_save_path:str = "",
                score_type:str='dsmvr',
                seed:int=2712
                )->None:
    
    @jit
    def vae_loss_grad(params:hk.Params, state_val:dict, rng_key:Array, x:Array):
        
        @jit
        def gaussian_likelihood(params:hk.Params):
            
            z, x_hat, mu, std = vae_apply_fn(params, x, rng_key, state_val)
            
            return jnp.mean(jnp.square(x-x_hat))
        
        @jit
        def kl1_fun(params:hk.Params):
            
            z, x_hat, mu, std = vae_apply_fn(params, x, rng_key, state_val)

            return z
                
        z, x_hat, mu, t = vae_apply_fn(params, x, rng_key, state_val)
        z_grad = grad(kl1_fun)(params)
        rec_grad = grad(gaussian_likelihood)(params)
        
        
        s_logqzx = vmap(lambda mu,z,t: score_apply_fn(score_state.params, jnp.hstack((mu,z,t)), 
                                  score_state.rng_key, score_state.state_val))(mu,z,t)
        s_logpz = vmap(lambda z: score_apply_fn(score_state.params, jnp.hstack((mu0,z,t0)), 
                                 rng_key, score_state.state_val))(z)
        
        kl_grad = jnp.mean(jnp.einsum('...i,...ij->....j', s_logqzx-s_logpz, z_grad), axis=0)

        return kl_grad-rec_grad
    
    @jit
    def vae_loss_fn(params:hk.Params, state_val:dict, rng_key:Array, x)->Array:
        
        @jit
        def gaussian_likelihood(x, x_hat):
            
            return jnp.mean(jnp.square(x-x_hat))
        
        @jit
        def kl_divergence(mu, std):
            
            return -0.5*jnp.mean(jnp.sum(1+2.0*std-mu**2-jnp.exp(2.0*std), axis=-1))
        
        z, x_hat, mu, t = vae_apply_fn(params, x, rng_key, state_val)
        std = t*jnp.ones(z.shape)
        
        rec_loss = gaussian_likelihood(x, x_hat)
        kld = kl_divergence(mu, std)
        
        elbo = rec_loss+kld
        
        return elbo
    
    @jit
    def score_loss_fn(params:hk.Params,  state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: score_apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:dim]
        xt = data[:,dim:(2*dim)]
        t = data[:,2*dim]
        dW = data[:,(2*dim+1):-1]
        dt = data[:,-1]
        
        return loss_model(score_generator, s1_model, params, state_val, rng_key,
                          x0, xt, t, dW, dt)
    
    @jit
    def update_vae_grad(state:TrainingState, data:Array):
        
        gradients = vae_loss_grad(state.params, state.state_val, state.rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key)
    
    @jit
    def update_vae_fun(state:TrainingState, data:Array):
        
        gradients = vae_loss_fun(state.params, state.state_val, state.rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        gradients = grad(score_loss_fn)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = vae_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if score_type == "vsm":
        loss_model = vsm_s1fun
    elif score_type == "dsm":
        loss_model = dsm_s1fun
    elif score_type == "dsmvr":
        loss_model = dsmvr_s1fun
    else:
        raise Exception("Invalid loss type. You can choose: vsm, dsm, dsmvr")
        return
        
    if vae_optimizer is None:
        vae_optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    if score_optimizer is None:
        score_optimizer = optax.adam(learning_rate = lr_rate,
                                     b1 = 0.9,
                                     b2 = 0.999,
                                     eps = 1e-08,
                                     eps_root = 0.0,
                                     mu_dtype=None)
        
        
    F = lambda z: decoder_apply_fn(vae_state.params, z[0].reshape(-1,2), vae_state.rng_key, 
                                   vae_state.state_val)[0].reshape(-1)
    M = Latent(dim=dim, emb_dim=emb_dim, F = F)
    x0 = (jnp.zeros(dim), jnp.zeros(1))
    score_generator = LocalSampling(M=M,
                                    x0=x0,
                                    repeats=8,
                                    x_samples=2**5,
                                    t_samples=2**7,
                                    N_sim=2**8,
                                    max_T=1.0,
                                    dt_steps=100,
                                    T_sample=0,
                                    t=0.1
                                    )
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(vae_model) == hk.Transformed:
        if vae_state is None:
            initial_params = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets))
            initial_opt_state = optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        if vae_state is None:
            initial_params, init_state = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets))
            initial_opt_state = optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, state_val, rng_key, data)[0]
        
    decoder_apply_fn = lambda params, data, rng_key, state_val: decoder_model.apply(params, rng_key, data)
        
    if type(score_model) == hk.Transformed:
        if score_state is None:
            initial_params = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            score_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, rng_key, data)
    elif type(score_model) == hk.TransformedWithState:
        if score_state is None:
            initial_params, init_state = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            score_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, state_val, rng_key, data)[0]
    
    F = lambda z: decoder_apply_fn(vae_state.params, z[0].reshape(-1,2), vae_state.rng_key, 
                                   vae_state.state_val)[0].reshape(-1)
    M = Latent(dim=dim, emb_dim=emb_dim, F = F)
    x0 = (jnp.zeros(dim), jnp.zeros(1))
    score_generator = LocalSampling(M=M,
                                    x0=x0,
                                    repeats=8,
                                    x_samples=2**5,
                                    t_samples=2**7,
                                    N_sim=2**8,
                                    max_T=1.0,
                                    dt_steps=100,
                                    T_sample=0,
                                    t=0.1
                                    )
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
    
    for step in range(burnin_epochs):
        for i in range(10):
            ds = next(vae_datasets)
            vae_state = update_vae_fun(vae_state, ds)
        if (step+1) % save_step == 0:
            save_model(vae_save_path, vae_state)
            print("Epoch: {}".format(step+1))
            
    F = lambda z: decoder_apply_fn(vae_state.params, z[0].reshape(-1,2), vae_state.rng_key, 
                                   vae_state.state_val)[0].reshape(-1)
    M = Latent(dim=dim, emb_dim=emb_dim, F = F)
    x0 = (jnp.zeros(dim), jnp.zeros(1))
    score_generator = LocalSampling(M=M,
                                    x0=x0,
                                    repeats=8,
                                    x_samples=2**5,
                                    t_samples=2**7,
                                    N_sim=2**8,
                                    max_T=1.0,
                                    dt_steps=100,
                                    T_sample=0,
                                    t=0.1
                                    )
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
            
    for step in range(burnin_epochs):
        data = next(score_datasets)
        if jnp.isnan(jnp.sum(data)):
            continue
        score_state = update_score(score_state, data)
        if (step+1) % save_step == 0:            
            save_model(score_save_path, score_state)
            print("Epoch: {}".format(step+1))
            
            
    for step in range(joint_epochs):
        for i in range(repeats):
            ds = next(vae_datasets)
            vae_state = update_vae_grad(vae_state, ds)
        F = lambda z: decoder_apply_fn(vae_state.params, z[0].reshape(-1,2), vae_state.rng_key, 
                                       vae_state.state_val)[0].reshape(-1)
        M = Latent(dim=dim, emb_dim=emb_dim, F = F)
        x0 = (jnp.zeros(dim), jnp.zeros(1))
        score_generator = LocalSampling(M=M,
                                        x0=x0,
                                        repeats=8,
                                        x_samples=2**5,
                                        t_samples=2**7,
                                        N_sim=2**8,
                                        max_T=1.0,
                                        dt_steps=100,
                                        T_sample=0,
                                        t=0.1
                                        )
        score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                       output_shapes=([batch_size,3*dim+2]))
        score_datasets = iter(tfds.as_numpy(score_datasets))
        for i in range(repeats):
            data = next(score_datasets)
            if jnp.isnan(jnp.sum(data)):
                continue
        if (step+1) % save_step == 0:
            
            save_model(vae_save_path, vae_state)
            save_model(score_save_path, score_state)
            print("Epoch: {}".format(step+1))
    
    save_model(vae_save_path, vae_state)
    save_model(score_save_path, score_state)
    print("Epoch: {}".format(step+1))
    
    return