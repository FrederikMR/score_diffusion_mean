#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:06:35 2023

@author: fmry
"""

#%% Sources

#%% Modules

#initialize modules
from src.setup import *
from src.params import *

#JaxGeometry 
from src.stochastics import product_sde
from src.stochastics.product_sde import tile

#adam optimizer
from jax.example_libraries.optimizers import adam

#haiku
import haiku as hk

#optax
import optax

#tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

#typing
from typing import Callable, NamedTuple

#%% Classes

class TrainingWithState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    
class TrainingNoState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

#%% Training Score

def train_full(M:object,
               model:object,
               data_generator:Callable[[], jnp.ndarray],
               update_coords:Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
               tmx_fun:Callable[[object, object, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
               N_dim:int,
               batch_size:int,
               dt:float,
               save_step:int=100,
               optimizer:Callable=None,
               opt_params:tuple=(0.1, 0.9, 0.999, 1e-8),
               file_path:str = "models/",
               loss_type:str='dsm',
               seed:int=2712
               )->None:
    
    def ApplyStateModel(state:object, data:jnp.ndarray)->jnp.ndarray:
        
        return model.apply(state.params,state.state_val, data)
    
    def ApplyNoStateModel(state:object, data:jnp.ndarray)->jnp.ndarray:
        
        return model.apply(state.params,data)
    
    def loss_vsm(state:object, data:jnp.ndarray)->float:
        """ compute loss."""
        s = apply_fn(state,data)
        norm2s = jnp.sum(s*s, axis=1)

        x0 = data[:,0:N_dim]
        xt = data[:,N_dim:2*N_dim]
        t = data[:,-1]
        
        (xts,chartts) = vmap(update_coords)(xt)
        
        divs = vmap(lambda x0, xt, chart, t: tmx_fun(model, state, x0, xt, chart, t))(x0,xts,chartts,t)
        
        return jnp.mean(norm2s+2.0*divs)

    def loss_dsm(state:object, data:jnp.ndarray)->float:
        
        def f(x): 
            x0 = data[0:N_dim]
            noise = data[N_dim:-1]
            
            loss = -noise/dt-apply_fn(state,jnp.hstack((x0, noise, t)))
            
            return jnp.sum(loss*loss)
        
        t = data[:,-1]
        
        loss = jnp.mean(vmap(
                    vmap(
                        f,
                        (0,0)),
                    (None, 1))(t,data[:,2*N_dim]))
    
        return loss
    
    def UpdateState(state, batch):
        
        gradients = grad(loss_fun)(state.params, batch)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, new_opt_state)
    
    def UpdateNoState(state, batch):
        
        gradients = grad(loss_fun)(state, batch)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingNoState(new_params, new_opt_state)
    
    if loss_type == "vsm":
        loss_fun = jit(loss_vsm)
    elif loss_type == "dsm":
        loss_fun = jit(loss_dsm)
    else:
        loss_fun = jit(loss_vsm)
        
    if optimizer is None:
        optimizer = adam
        opt_init, opt_update, get_params = optimizer(0.1, b1=0.9, b2=0.999, eps=1e-8)
    else:
        opt_init, opt_update, get_params = optimizer(*opt_params)
        
    train_dataset = tf.data.Dataset.from_generator(data_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,2*N_dim+1]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
    
    if type(model) == hk.Transformed:
        initial_params = model.init(random.PRNGKey(seed), next(train_dataset))
        initial_opt_state = optimizer.init(initial_params)
        state = TrainingNoState(initial_params, initial_opt_state)
        update = jit(UpdateNoState)
        apply_fn = jit(ApplyNoStateModel)
    elif type(model) == hk.TransformedWithState:
        initial_params, init_state = model.init(random.PRNGKey(seed), next(train_dataset))
        initial_opt_state = optimizer.init(initial_params)
        state = TrainingWithState(initial_params, init_state, initial_opt_state)
        update = jit(UpdateState)
        apply_fn = jit(ApplyStateModel)
    
    loss = []
    for step in range(epochs):
        state = update(state, next(train_dataset))
        if step % save_step == 0:
            loss_val = loss_fun(state.params, next(train_dataset))
            loss_val = jax.device_get(loss_val).item()
            loss.append(loss_val)
            
            file_name = os.path.join(save_path, "loss_arrays.npy")
            np.save(file_name, jnp.stack(loss))
            
            save_model_fn(save_path, state, step)
            print("Epoch: {} \t loss = {:.4f}".format(step, loss_val))

    file_name = os.path.join(save_path, "loss_arrays.npy")
    np.save(file_name, jnp.stack(loss))
    
    save_model_fn(save_path, state, epochs)
    
    return