#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:06:35 2023

@author: fmry
"""

#%% Sources

#%% Modules

#initialize modules
from JAXGeometry.setup import *
from JAXGeometry.params import *

#JaxGeometry 
from JAXGeometry.stochastics import product_sde
from JAXGeometry.stochastics.product_sde import tile
from JAXGeometry.statistics.score_matching.model_loader import save_model

#adam optimizer
from jax.example_libraries.optimizers import adam

#haiku
import haiku as hk

#optax
import optax

#tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
#%% Classes

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    rng_key: Array
    
#%% Training Score

def train_full(M:object,
               model:object,
               data_generator:Callable[[], jnp.ndarray],
               update_coords:Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
               tmx_fun:Callable[[hk.Params, dict, Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
               N_dim:int,
               batch_size:int,
               dt:float,
               epochs:int=100,
               save_step:int=100,
               optimizer:Callable=None,
               opt_params:tuple=(0.1, 0.9, 0.999, 1e-8),
               save_path:str = "",
               loss_type:str='vsm',
               seed:int=2712
               )->None:
    
    def loss_vsm(params:hk.Params, state_val:dict, rng_key:Array, data:jnp.ndarray)->float:
        """ compute loss."""
        s = apply_fn(params, data, rng_key, state_val)
        norm2s = jnp.sum(s*s, axis=1)

        x0 = data[:,0:N_dim]
        xt = data[:,N_dim:2*N_dim]
        t = data[:,-1]
        
        (xts,chartts) = vmap(update_coords)(xt)
        
        divs = vmap(lambda x0, xt, chart, t: 
                    M.div((xt, chart), lambda x: tmx_fun(params, state_val, rng_key, x0, xt, chart, t)))(x0,xts,chartts,t)
        
        return jnp.mean(norm2s+2.0*divs)

    def loss_dsm(params:hk.Params, state_val:dict, rng_key:Array, data:jnp.ndarray)->float:
        
        def f(x): 
            x0 = data[0:N_dim]
            noise = data[N_dim:-1]
            
            loss = -noise/dt-apply_fn(params,jnp.hstack((x0, noise, t)), rng_key, state_val)
            
            return jnp.sum(loss*loss)
        
        t = data[:,-1]
        
        loss = jnp.mean(vmap(
                    vmap(
                        f,
                        (0,0)),
                    (None, 1))(t,data[:,2*N_dim]))
    
        return loss
    
    @jit
    def update(state:TrainingState, data:jnp.ndarray):
        
        rng_key, next_rng_key = jax.random.split(state.rng_key)
        gradients, loss = jax.value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
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
    
    initial_rng_key = jax.random.PRNGKey(seed)
    if type(model) == hk.Transformed:
        initial_params = model.init(jax.random.PRNGKey(seed), next(train_dataset))
        initial_opt_state = optimizer.init(initial_params)
        state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        initial_params, init_state = model.init(jax.random.PRNGKey(seed), next(train_dataset))
        initial_opt_state = optimizer.init(initial_params)
        state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        state, loss_val = update(state, next(train_dataset))
        if step % save_step == 0:
            loss_val = jax.device_get(loss_val).item()
            loss.append(loss_val)
            
            np.save(os.path.join(save_path, "loss_arrays.npy"), jnp.stack(loss))
            
            save_model(save_path, state)
            print("Epoch: {} \t loss = {:.4f}".format(step+1, loss_val))

    loss_val = jax.device_get(loss_val).item()
    loss.append(loss_val)
    
    np.save(os.path.join(save_path, "loss_arrays.npy"), jnp.stack(loss))
    
    save_model(save_path, state)
    print("Epoch: {} \t loss = {:.4f}".format(step+1, loss_val))
    
    return