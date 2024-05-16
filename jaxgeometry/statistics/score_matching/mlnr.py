#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:38:42 2023

@author: fmry
"""

#%% Sources

#https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670

#%% Modules

from jaxgeometry.setup import *
from .model_loader import save_model

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    rng_key: Array

#%% Maximumu Likelihood Neural Regression

def train_mlnr(input_data:Array,
               output_data:Array,
               M:object,
               model:object,
               grady_log:Callable,
               gradt_log:Callable,
               state:TrainingState = None,
               lr_rate:float = 0.001,
               batch_size:int=100,
               epochs:int=100,
               warmup_epochs:int=1000,
               save_step:int=100,
               optimizer:object=None,
               save_path:str = "",
               min_t:float=1e-2,
               max_t:float=1.0,
               seed:int=2712
               )->None:
    
    def learning_rate_fn():
        """Creates learning rate schedule."""
        warmup_fn = optax.linear_schedule(
            init_value=.0, end_value=lr_rate,
            transition_steps=warmup_epochs)
        cosine_epochs = max(epochs - warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=lr_rate,
            decay_steps=epochs - warmup_epochs)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_epochs])
        return schedule_fn
    
    @jit
    def loss_fun(params:hk.Params, state_val:dict, rng_key:Array, data:Array):
        
        neural_model = lambda x: apply_fn(params, x, rng_key, state_val)
        x,y = data
        
        f_data, sigma_data = neural_model(x)
        sigma2 = jnp.clip(sigma_data**2, min_t, max_t)
        
        loss1 = jnp.einsum('...i,...i->...', 
                           grady_log(y,lax.stop_gradient(f_data), 
                                     lax.stop_gradient(sigma2)),
                           f_data)
        loss2 = gradt_log(y,lax.stop_gradient(f_data),
                          lax.stop_gradient(sigma2))*sigma2
        
        return -jnp.sum(loss1+loss2)
    
    @jit
    def update(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
        
    lr_schedule = learning_rate_fn()
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_schedule,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    ds = tf.data.Dataset.from_tensor_slices((input_data, 
                                             output_data)).shuffle(buffer_size=100).batch(batch_size).repeat(epochs)
    train_dataset = iter(tfds.as_numpy(ds))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(model) == hk.Transformed:
        if state is None:
            initial_params = model.init(jrandom.PRNGKey(seed), next(train_dataset)[0])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = model.init(jrandom.PRNGKey(seed), next(train_dataset)[0])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        state, loss_val = update(state, data)
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