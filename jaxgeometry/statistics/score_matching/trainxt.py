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
def train_s1(M:object,
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
             loss_type:str='dsmvr',
             seed:int=2712
             )->None:
    
    @jit
    def loss_fun(params:hk.Params, state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:(2*N_dim)]
        t = data[:,2*N_dim]
        dW = data[:,(2*N_dim+1):-1]
        dt = data[:,-1]
        
        return loss_model(generator, s1_model, x0, xt, t, dW, dt)
    
    @jit
    def update(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if loss_type == "vsm":
        loss_model = vsm_s1fun
    elif loss_type == "dsm":
        loss_model = dsm_s1fun
    elif loss_type == "dsmvr":
        loss_model = dsmvr_s1fun
    else:
        raise Exception("Invalid loss type. You can choose: vsm, dsm, dsmvr")
        
        return
        
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,2*N_dim+dW_dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(model) == hk.Transformed:
        if state is None:
            initial_params = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        if ((jnp.isnan(jnp.sum(data)))):
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,2*N_dim+dW_dim+2]))
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

#%% Score Training for SN
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_s2(M:object,
             s1_model:Callable[[Array, Array, Array], Array],
             s2_model:object,
             generator:object,
             N_dim:int,
             dW_dim:int,
             batch_size:int,
             state:TrainingState=None,
             lr_rate:float=0.0002,
             epochs:int=100,
             save_step:int=100,
             optimizer:object=None,
             save_path:str = "",
             seed:int=2712,
             loss_type:str = "dsmvr",
             )->None:
    
    @jit
    def loss_fun(params:hk.Params, 
                 state_val:dict, 
                 rng_key:Array, 
                 data:Array
                 )->float:
        
        s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:(2*N_dim)]
        t = data[:,2*N_dim]
        dW = data[:,(2*N_dim+1):-1]
        dt = data[:,-1]
        
        return loss_model(generator, s1_model, s2_model, x0, xt, t, dW, dt)
    
    @jit
    def update(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if loss_type == 'dsm':
        loss_model = dsm_s2fun
    elif loss_type == "dsmdiag":
        loss_model = dsmdiag_s2fun
    elif loss_type == "dsmvr":
        loss_model = dsmvr_s2fun
    elif loss_type == "dsmdiagvr":
        loss_model = dsmdiagvr_s2fun
        
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,2*N_dim+dW_dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
    
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(s2_model) == hk.Transformed:
        if state is None:
            initial_params = s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s2_model.apply(params, rng_key, data)
    elif type(s2_model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s2_model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        if ((jnp.isnan(jnp.sum(data)))):
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,2*N_dim+dW_dim+2]))
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

#%% Score Training for SN
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_s1s2(M:object,
               s1s2_model:object,
               generator:object,
               N_dim:int,
               dW_dim:int,
               batch_size:int,
               state:TrainingState=None,
               s1_params:dict=None,
               s2_params:dict=None,
               lr_rate:float=0.0002,
               epochs:int=100,
               save_step:int=100,
               optimizer:object=None,
               gamma:float = 1.0,
               save_path:str = "",
               seed:int=2712,
               loss_type:str='dsmvr'
               )->None:
    
    @jit
    def loss_fun(params:hk.Params, 
                 state_val:dict, 
                 rng_key:Array, 
                 data:Array
                 )->float:
        
        s1_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)[0]
        s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)[1]
    
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:(2*N_dim)]
        t = data[:,2*N_dim]
        dW = data[:,(2*N_dim+1):-1]
        dt = data[:,-1]
        
        s1_loss = loss_s1model(generator, s1_model, x0, xt, t, dW, dt)
        s2_loss = loss_s2model(generator, s1_model, s2_model, x0, xt, t, dW, dt)
        
        return s2_loss+gamma*s1_loss
    
    @jit
    def update(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if loss_type == 'dsm':
        loss_s1model = dsm_s1fun
        loss_s2model = dsm_s2fun
    elif loss_type == "dsmdiag":
        loss_s1model = dsm_s1fun
        loss_s2model = dsmdiag_s2fun
    elif loss_type == "dsmvr":
        loss_s1model = dsmvr_s1fun
        loss_s2model = dsmvr_s2fun
    elif loss_type == "dsmdiagvr":
        loss_s1model = dsmvr_s1fun
        loss_s2model = dsmdiagvr_s2fun
        
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,2*N_dim+dW_dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
    
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(s1s2_model) == hk.Transformed:
        if state is None:
            initial_params = s1s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            if s1_params is not None:
                initial_params.update(s1_params)
            if s2_params is not None:
                initial_params.update(s2_params)
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s1s2_model.apply(params, rng_key, data)
    elif type(s1s2_model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = s1s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            if s1_params is not None:
                initial_params.update(s1_params)
            if s2_params is not None:
                initial_params.update(s2_params)
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s1s2_model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        if ((jnp.isnan(jnp.sum(data)))):
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,2*N_dim+dW_dim+2]))
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