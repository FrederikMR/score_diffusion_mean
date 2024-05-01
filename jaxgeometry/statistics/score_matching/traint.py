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

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    rng_key: Array

#%% Score Training
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_t(M:object,
            model:object,
            generator:object,
            N_dim:int,
            dW_dim:int,
            batch_size:int,
            state:TrainingState = None,
            lr_rate:float = 0.001,
            epochs:int=100,
            save_step:int=100,
            optimizer:object=None,
            save_path:str = "",
            seed:int=2712
            )->None:
    
    @jit
    def loss_fun(params:hk.Params, state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:,:N_dim]
        xt = data[:,:,N_dim:(2*N_dim)]
        t = data[:,:,2*N_dim]
        #dW = data[:,:,(2*N_dim+1):-1]
        #dt = data[:,:,-1]
        
        x0_t1, x0_t2 = x0[0], x0[-1]
        xt_t1, xt_t2 = xt[0], xt[-1]
        t1, t2 = t[0], t[-1]
        
        loss_s1 = vmap(lambda x1,x2,t1: vmap(lambda x,y,t: s1_model(x,y,t))(x1,x2,t1))(x0,xt,t)
        loss_t1 = vmap(lambda x,y,t: s1_model(x, y, t))(x0_t1, xt_t1, t1)
        loss_t2 = vmap(lambda x,y,t: s1_model(x, y, t))(x0_t2, xt_t2, t2)
        loss_dt = vmap(lambda x1,x2,t1: \
                       vmap(lambda x,y,t: jacfwd(lambda t0: s1_model(x,y,t0))(t).squeeze())(x1,x2,t1))(x0,xt,t)
        
        term1 = jnp.mean(jnp.mean(loss_s1*loss_s1+2.0*loss_dt, axis=-1))
        term2 = 2.0*jnp.mean(loss_t1-loss_t2)
        
        return term1+term2

    @jit
    def update(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
        
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([generator.dt_steps,
                                                                   generator.N_sim,
                                                                   2*N_dim+dW_dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(model) == hk.Transformed:
        if state is None:
            initial_params = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        if ((jnp.isnan(jnp.sum(data)))):
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.dt_steps,
                                                                           generator.N_sim,
                                                                           2*N_dim+dW_dim+2]))
            train_dataset = iter(tfds.as_numpy(train_dataset))
            continue
        new_state, loss_val = update(state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in new_state.params[list(new_state.params.keys())[0]].values())) \
                and (loss_val<1e12)):
            state = new_state
        else:
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,2*N_dim+dW_dim+2]))
            train_dataset = iter(tfds.as_numpy(train_dataset))
        if (step+1) % save_step == 0:
            loss_val = device_get(loss_val).item()
            loss.append(loss_val)
            
            np.save(os.path.join(save_path, "loss_arrays.npy"), jnp.stack(loss))
            
            save_model(save_path, state)
            print("Epoch: {} \t loss = {:.4f}".format(step+1, loss_val))

    loss.append(loss_val)
    
    np.save(os.path.join(save_path, "loss_arrays.npy"), jnp.stack(loss))
    
    save_model(save_path, state)
    print("Epoch: {} \t loss = {:.4f}".format(step+1, loss_val))
    
    return
