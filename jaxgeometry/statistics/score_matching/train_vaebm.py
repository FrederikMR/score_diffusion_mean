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
from .generators import LocalSampling, VAESampling
from jaxgeometry.manifolds import Latent
from .vae_loss_fun import vae_euclidean_loss, vae_riemannian_loss

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    rng_key: Array
    
#%% Pre-Train VAE

def pretrain_vae(vae_model:object,
                 data_generator:object,
                 lr_rate:float = 0.002,
                 save_path:str = '',
                 split:float = 0.0,
                 vae_state:TrainingState = None,
                 epochs:int=1000,
                 save_step:int = 100,
                 vae_optimizer:object = None,
                 seed:int=2712,
                 )->None:
    
    @partial(jit, static_argnames=['training_type'])
    def update(state:TrainingState, data:Array, training_type="All"):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        gradients = grad(vae_euclidean_loss)(state.params, state.state_val, state.rng_key, data,
                                             vae_apply_fn, training_type=training_type)
        updates, new_opt_state = vae_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key)
    
    if vae_optimizer is None:
        vae_optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    if type(vae_model) == hk.Transformed:
        if vae_state is None:
            initial_params = vae_model.init(jrandom.PRNGKey(seed), next(data_generator))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        if vae_state is None:
            initial_params, init_state = vae_model.init(jrandom.PRNGKey(seed), next(data_generator))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, state_val, rng_key, data)[0]
    
    if split>0:
        epochs_encoder = int(split*epochs)
        epochs_decoder = int((1-split)*epochs)
        epochs = 0
    else:
        epochs_encoder = 0
        epochs_decoder = 0
    
    for step in range(epochs_encoder):
        ds = next(data_generator)
        vae_state = update(vae_state, ds, training_type="Encoder")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print("Epoch: {}".format(step+1))
    for step in range(epochs_decoder):
        ds = next(data_generator)
        vae_state = update(vae_state, ds, training_type="Decoder")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print("Epoch: {}".format(step+1))
    
    for step in range(epochs):
        ds = next(data_generator)
        vae_state = update(vae_state, ds, training_type="All")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print("Epoch: {}".format(step+1))
          
    save_model(save_path, vae_state)
    
    return

#%% Pre-Train Scores

def pretrain_scores(score_model:object,
                    vae_state:object,
                    decoder_model:object,
                    data_generator:object,
                    x0s:Array,
                    lr_rate:float = 0.002,
                    save_path:str = '',
                    repeats:int=8,
                    x_samples:int=2**5,
                    t_samples:int=2**7,
                    N_sim:int=2**8,
                    max_T:float=1.0,
                    dt_steps:int=1000,
                    training_type:str = 'dsmvr',
                    score_state:TrainingState = None,
                    epochs:int=1000,
                    save_step:int = 100,
                    score_optimizer:object = None,
                    seed:int=2712,
                    ):
    
    @jit
    def loss_fun(params:hk.Params,  state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: score_apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:dim]
        xt = data[:,dim:(2*dim)]
        t = data[:,2*dim]
        dW = data[:,(2*dim+1):-1]
        dt = data[:,-1]
        
        return loss_model(score_generator, s1_model, params, state_val, rng_key,
                          x0, xt, t, dW, dt)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, state.opt_state)
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
    
    initial_rng_key = jrandom.PRNGKey(seed)
    if score_optimizer is None:
        score_optimizer = optax.adam(learning_rate = lr_rate,
                                     b1 = 0.9,
                                     b2 = 0.999,
                                     eps = 1e-08,
                                     eps_root = 0.0,
                                     mu_dtype=None)
        
    batch_size = x_samples*repeats*t_samples
    dim = x0s.shape[-1]
    F = lambda z: decoder_model.apply(vae_state.params, vae_state.rng_key, z.reshape(-1,dim))

    score_generator = VAESampling(F=F,
                                  dim=dim,
                                  x0=x0s,
                                  repeats=repeats,
                                  x_samples=x_samples,
                                  t_samples=t_samples,
                                  N_sim=N_sim,
                                  max_T=max_T,
                                  dt_steps=dt_steps,
                                  )
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
    
    if type(score_model) == hk.Transformed:
        if score_state is None:
            initial_params = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, rng_key, data)
    elif type(score_model) == hk.TransformedWithState:
        if score_state is None:
            initial_params, init_state = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, state_val, rng_key, data)[0]
        
    for step in range(epochs):
        data = next(score_datasets)
        if jnp.isnan(jnp.sum(data)):
            score_generator.x0s = score_generator.x0s_default
            score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,3*dim+2]))
            score_datasets = iter(tfds.as_numpy(score_datasets))
            continue
        score_new_state,loss = update_score(score_state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in score_new_state.params[list(score_new_state.params.keys())[0]].values())) \
                and (loss<1e12)):
            score_state = score_new_state
        else:
            score_generator.x0s = score_generator.x0s_default
            score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,3*dim+2]))
            score_datasets = iter(tfds.as_numpy(score_datasets))
        if (step+1) % save_step == 0:            
            save_model(save_path, score_state)
            print("Epoch: {}".format(step+1))
    
    return

#%% Score Training

def train_vaebm(vae_model:object,
                decoder_model:object,
                score_model:object,
                vae_datasets:object,
                dim:int,
                epochs:int=1000,
                vae_epochs:int=100,
                score_epochs:int=100,
                vae_split:float=0.0,
                lr_rate_vae:float=0.002,
                lr_rate_score:float=0.002,
                score_repeats:int=8,
                score_x_samples:int=2**5,
                score_t_samples:int=2**7,
                score_N_sim:int=2**8,
                max_T:float=1.0,
                dt_steps:int=1000,
                vae_optimizer:object = None,
                score_optimizer:object = None,
                vae_state:TrainingState=None,
                score_state:TrainingState=None,
                seed:int=2712,
                save_step:int=100,
                score_type:str='dsmvr',
                vae_path:str='',
                score_path:str='',
                )->None:
    
    @partial(jit, static_argnames=['training_type'])
    def update(state:TrainingState, data:Array, training_type="All"):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        gradients = grad(vae_riemannian_loss)(state.params, state.state_val, state.rng_key, data,
                                             vae_apply_fn, score_apply_fn, score_state, 
                                             training_type=training_type)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key)
    
    @jit
    def score_loss(params:hk.Params,  state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: score_apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:dim]
        xt = data[:,dim:(2*dim)]
        t = data[:,2*dim]
        dW = data[:,(2*dim+1):-1]
        dt = data[:,-1]
        
        return loss_model(score_generator, s1_model, params, state_val, rng_key,
                          x0, xt, t, dW, dt)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(score_loss)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, state.opt_state)
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
        vae_optimizer = optax.adam(learning_rate = lr_rate_vae,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    if score_optimizer is None:
        score_optimizer = optax.adam(learning_rate = lr_rate_score,
                                     b1 = 0.9,
                                     b2 = 0.999,
                                     eps = 1e-08,
                                     eps_root = 0.0,
                                     mu_dtype=None)
        
    F = lambda z: decoder_apply_fn(vae_state.params, z.reshape(-1,dim), vae_state.rng_key, 
                                   vae_state.state_val)[0].reshape(-1)
    x0 = jnp.zeros(dim)
    score_generator = VAESampling(F=F,
                                  dim=dim,
                                  x0=x0,
                                  repeats=score_repeats,
                                  x_samples=score_x_samples,
                                  t_samples=score_t_samples,
                                  N_sim=score_N_sim,
                                  max_T=max_T,
                                  dt_steps=dt_steps,
                                  )
    batch_size = score_x_samples*score_repeats*score_t_samples
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(vae_model) == hk.Transformed:
        if vae_state is None:
            initial_params = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        if vae_state is None:
            initial_params, init_state = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, state_val, rng_key, data)[0]
        
    decoder_apply_fn = lambda params, data, rng_key, state_val: decoder_model.apply(params, rng_key, data)
        
    if type(score_model) == hk.Transformed:
        if score_state is None:
            initial_params = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, rng_key, data)
    elif type(score_model) == hk.TransformedWithState:
        if score_state is None:
            initial_params, init_state = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, state_val, rng_key, data)[0]
        
    if vae_split>0:
        epochs_encoder = int(vae_split*vae_epochs)
        epochs_decoder = int((1-vae_split)*vae_epochs)
        vae_epochs = 0
    else:
        epochs_encoder = 0
        epochs_decoder = 0
    
    for step in range(epochs):
        for vae_step in range(epochs_encoder):
            ds = next(vae_datasets)
            vae_state = update(vae_state, ds, training_type="Encoder")
        for vae_step in range(epochs_decoder):
            ds = next(vae_datasets)
            vae_state = update(vae_state, ds, training_type="Decoder")
        for vae_step in range(vae_epochs):
            ds = next(vae_datasets)
            vae_state = update(vae_state, ds, training_type="All")

        F = lambda z: decoder_apply_fn(vae_state.params, z.reshape(-1,dim), vae_state.rng_key, 
                                       vae_state.state_val)[0].reshape(-1)
        z, *_ = vae_apply_fn(vae_state.params, ds, vae_state.rng_key, vae_state.state_val)
        score_generator.F = F
        z0 = z[np.round(np.linspace(0, len(z) - 1, score_repeats)).astype(int)]
        score_generator.x0s = z0
        score_generator.x0s_default = z0
        score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                       output_shapes=([batch_size,3*dim+2]))
        score_datasets = iter(tfds.as_numpy(score_datasets))
        
        for score_step in range(score_epochs):
            data = next(score_datasets)
            if jnp.isnan(jnp.sum(data)):
                score_generator.x0s = score_generator.x0s_default
                score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                               output_shapes=([batch_size,3*dim+2]))
                score_datasets = iter(tfds.as_numpy(score_datasets))
                continue
            score_new_state,loss = update_score(score_state, data)
            if ((not any(jnp.sum(jnp.isnan(val))>0 for val in score_new_state.params[list(score_new_state.params.keys())[0]].values())) \
                    and (loss<1e12)):
                score_state = score_new_state
            else:
                score_generator.x0s = score_generator.x0s_default
                score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                               output_shapes=([batch_size,3*dim+2]))
                score_datasets = iter(tfds.as_numpy(score_datasets))
                
        if (step+1) % save_step == 0:            
            save_model(score_path, score_state)
            save_model(vae_path, vae_state)
            print("Epoch: {}".format(step+1))
          
    save_model(score_path, score_state)
    save_model(vae_path, vae_state)
    
    return

#%% Score Training
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_vaebm_old(vae_model:object,
                decoder_model:object,
                score_model:object,
                vae_datasets:object,
                dim:int,
                emb_dim:int,
                vae_state:TrainingState = None,
                score_state:TrainingState = None,
                lr_rate:float = 0.001,
                burnin_epochs:int=100,
                joint_epochs:int=100,
                repeats:int=1,
                save_step:int=100,
                score_repeats:int=8,
                x_samples:int=2**5,
                t_samples:int=2**7,
                N_sim:int=2**8,
                max_T:float=1.0,
                dt_steps:int=1000,
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
            
            _, x_hat, *_ = vae_apply_fn(params, x, rng_key, state_val)
            
            return jnp.mean(jnp.square(x-x_hat))
        
        @jit
        def kl1_fun(params:hk.Params):
            
            z, *_ = vae_apply_fn(params, x, rng_key, state_val)

            return z
                
        z, x_hat, mu, t, mu0, t0 = vae_apply_fn(params, x, rng_key, state_val)
        z_grad = jacfwd(kl1_fun)(params)
        rec_grad = grad(gaussian_likelihood)(params)
        
        s_logqzx = vmap(lambda mu,z,t: score_apply_fn(score_state.params, jnp.hstack((mu,z,t)), 
                                  score_state.rng_key, score_state.state_val))(mu,z,t)
        s_logpz = vmap(lambda mu0,z,t0: score_apply_fn(score_state.params, jnp.hstack((mu0,z,t0)), 
                                 rng_key, score_state.state_val))(mu0,z,t0)
        diff = s_logqzx-s_logpz

        if x.ndim == 1:
            kl_grad = {layer: {name: jnp.einsum('i,i...->...', diff, w) for name,w in val.items()} \
                       for layer,val in z_grad.items()}
        else:
            kl_grad = {layer: {name: jnp.mean(jnp.einsum('ki,ki...->k...', diff, w), axis=0) for name,w in val.items()} \
                       for layer,val in z_grad.items()}
        #z_grad.update((x, jnp.einsum('...i,...ij->...j')) for v,w in x for x, y in my_dict.items())
        #kl_grad = jnp.mean(jnp.einsum('...i,...ij->...j', s_logqzx-s_logpz, z_grad), axis=0)
        
        res = {layer: {name: kl_grad[layer][name]-rec_grad[layer][name] for name,w in val.items()} \
                   for layer,val in z_grad.items()}

        return res
    
    @jit
    def vae_loss_fn(params:hk.Params, state_val:dict, rng_key:Array, x)->Array:
        
        @jit
        def gaussian_likelihood(x, x_hat):
            
            return jnp.mean(jnp.square(x-x_hat))
        
        @jit
        def kl_divergence(mu, std):
            
            return -0.5*jnp.mean(jnp.sum(1+2.0*std-mu**2-jnp.exp(2.0*std), axis=-1))
        
        z, x_hat, mu, t, mu0, t0 = vae_apply_fn(params, x, rng_key, state_val)
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
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        gradients = vae_loss_grad(state.params, state.state_val, state.rng_key, data)
        updates, new_opt_state = vae_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key)
    
    @jit
    def update_vae_fun(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        gradients = grad(vae_loss_fn)(state.params, state.state_val, state.rng_key, data)
        updates, new_opt_state = vae_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(score_loss_fn)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, state.opt_state)
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
        
        
    F = lambda z: decoder_apply_fn(vae_state.params, z.reshape(-1,dim), vae_state.rng_key, 
                                   vae_state.state_val)[0].reshape(-1)
    x0 = jnp.zeros(dim)
    score_generator = VAESampling(F=F,
                                  dim=dim,
                                  x0=x0,
                                  repeats=8,
                                  x_samples=2**5,
                                  t_samples=2**7,
                                  N_sim=2**8,
                                  max_T=1.0,
                                  dt_steps=1000,
                                  )
    batch_size = (2**5)*(2**3)*(2**7)
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(vae_model) == hk.Transformed:
        if vae_state is None:
            initial_params = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        if vae_state is None:
            initial_params, init_state = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, state_val, rng_key, data)[0]
        
    decoder_apply_fn = lambda params, data, rng_key, state_val: decoder_model.apply(params, rng_key, data)
        
    if type(score_model) == hk.Transformed:
        if score_state is None:
            initial_params = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, rng_key, data)
    elif type(score_model) == hk.TransformedWithState:
        if score_state is None:
            initial_params, init_state = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, state_val, rng_key, data)[0]
    
    F = lambda z: decoder_apply_fn(vae_state.params, z.reshape(-1,dim), vae_state.rng_key, 
                                   vae_state.state_val)[0].reshape(-1)

    x0 = jnp.zeros(dim)
    score_generator = VAESampling(F=F,
                                  dim=dim,
                                  x0=x0,
                                  repeats=8,
                                  x_samples=2**5,
                                  t_samples=2**7,
                                  N_sim=2**8,
                                  max_T=1.0,
                                  dt_steps=1000,
                                  )
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
    
    for step in range(burnin_epochs):
        ds = next(vae_datasets)
        vae_state = update_vae_fun(vae_state, ds)
        if (step+1) % save_step == 0:
            save_model(vae_save_path, vae_state)
            print("Epoch: {}".format(step+1))
            
    z, *_ = vae_apply_fn(vae_state.params, ds, vae_state.rng_key, vae_state.state_val)
    F = lambda z: decoder_apply_fn(vae_state.params, z.reshape(-1,dim), vae_state.rng_key, 
                                   vae_state.state_val).reshape(-1)
    score_generator.F = F
    idx = np.round(np.linspace(0, len(z) - 1, 8)).astype(int)
    x0 = z[idx]
    score_generator.x0s = x0
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
            
    for step in range(burnin_epochs):
        data = next(score_datasets)
        if jnp.isnan(jnp.sum(data)):
            score_generator.x0s = score_generator.x0s_default
            score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,3*dim+2]))
            score_datasets = iter(tfds.as_numpy(score_datasets))
            continue
        score_new_state,loss = update_score(score_state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in score_new_state.params[list(score_new_state.params.keys())[0]].values())) \
                and (loss<1e12)):
            score_state = score_new_state
        else:
            score_generator.x0s = score_generator.x0s_default
            score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,3*dim+2]))
            score_datasets = iter(tfds.as_numpy(score_datasets))
        if (step+1) % save_step == 0:            
            save_model(score_save_path, score_state)
            print("Epoch: {}".format(step+1))
            
    for step in range(joint_epochs):
        for i in range(repeats):
            ds = next(vae_datasets)
            vae_state = update_vae_grad(vae_state, ds)
        F = lambda z: decoder_apply_fn(vae_state.params, z[0].reshape(-1,dim), vae_state.rng_key, 
                                       vae_state.state_val).reshape(-1)
        z, *_ = vae_apply_fn(vae_state.params, ds, vae_state.rng_key, vae_state.state_val)
        idx = np.round(np.linspace(0, len(z) - 1, 8)).astype(int)
        x0 = z[idx]
        #x0 = (z[::repeats], jnp.zeros(repeats))
        score_generator.x0s = x0
        score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                       output_shapes=([batch_size,3*dim+2]))
        score_datasets = iter(tfds.as_numpy(score_datasets))
        for i in range(repeats):
            data = next(score_datasets)
            if jnp.isnan(jnp.sum(data)):
                continue
            score_state,loss = update_score(score_state, data)
            print(loss)
        if (step+1) % save_step == 0:
            
            save_model(vae_save_path, vae_state)
            save_model(score_save_path, score_state)
            print("Epoch: {}".format(step+1))
    
    save_model(vae_save_path, vae_state)
    save_model(score_save_path, score_state)
    print("Epoch: {}".format(step+1))
    
    return