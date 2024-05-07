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
             state:TrainingState = None,
             lr_rate:float = 0.001,
             epochs:int=100,
             warmup_epochs:int=1000,
             save_step:int=100,
             optimizer:object=None,
             save_path:str = "",
             loss_type:str='dsmvr',
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
        
        s1_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:,:generator.dim].reshape(-1, generator.dim)
        xt = data[:,:,generator.dim:(2*generator.dim)].reshape(-1,generator.dim)
        t = data[:,:,2*generator.dim].reshape(-1)
        dW = data[:,:,(2*generator.dim+1):-1].reshape(-1,generator.dim)
        dt = data[:,:,-1].reshape(-1)
        
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
    elif loss_type == "ssm":
        loss_model = ssm_s1fun()
    elif loss_type == "ssmvr":
        loss_model = ssmvr_s1fun()
    else:
        raise Exception("Invalid loss type. You can choose: vsm, dsm, dsmvr")
        
        return
        
    lr_schedule = learning_rate_fn()
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_schedule,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([generator.t_samples,
                                                                   generator.N_sim,
                                                                   3*generator.dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(model) == hk.Transformed:
        if state is None:
            initial_params = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        if ((jnp.isnan(jnp.sum(data)))):
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.t_samples,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
            train_dataset = iter(tfds.as_numpy(train_dataset))
            continue
        new_state, loss_val = update(state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in new_state.params[list(new_state.params.keys())[0]].values())) \
                and (loss_val<1e12)):
            state = new_state
        else:
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.t_samples,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
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
             state:TrainingState=None,
             lr_rate:float=0.0002,
             epochs:int=100,
             warmup_epochs:int=1000,
             save_step:int=100,
             optimizer:object=None,
             save_path:str = "",
             seed:int=2712,
             loss_type:str = "dsmvr",
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
    def loss_fun(params:hk.Params, 
                 state_val:dict, 
                 rng_key:Array, 
                 data:Array
                 )->float:
        
        s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:,:generator.dim].reshape(-1, generator.dim)
        xt = data[:,:,generator.dim:(2*generator.dim)].reshape(-1,generator.dim)
        t = data[:,:,2*generator.dim].reshape(-1)
        dW = data[:,:,(2*generator.dim+1):-1].reshape(-1,generator.dim)
        dt = data[:,:,-1].reshape(-1)
        
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
        
    lr_schedule = learning_rate_fn()
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_schedule,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([generator.t_samples,
                                                                   generator.N_sim,
                                                                   3*generator.dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
    
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(s2_model) == hk.Transformed:
        if state is None:
            initial_params = s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s2_model.apply(params, rng_key, data)
    elif type(s2_model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s2_model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        if ((jnp.isnan(jnp.sum(data)))):
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.t_samples,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
            train_dataset = iter(tfds.as_numpy(train_dataset))
            continue
        new_state, loss_val = update(state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in new_state.params[list(new_state.params.keys())[0]].values())) \
                and (loss_val<1e12)):
            state = new_state
        else:
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.t_samples,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
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
               state:TrainingState=None,
               s1_params:dict=None,
               s2_params:dict=None,
               lr_rate:float=0.0002,
               epochs:int=100,
               warmup_epochs:int=1000,
               save_step:int=100,
               optimizer:object=None,
               gamma:float = 1.0,
               save_path:str = "",
               seed:int=2712,
               loss_type:str='dsmvr'
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
    def loss_fun(params:hk.Params, 
                 state_val:dict, 
                 rng_key:Array, 
                 data:Array
                 )->float:
        
        s1_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)[0]
        s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)[1]
    
        x0 = data[:,:,:generator.dim].reshape(-1, generator.dim)
        xt = data[:,:,generator.dim:(2*generator.dim)].reshape(-1,generator.dim)
        t = data[:,:,2*generator.dim].reshape(-1)
        dW = data[:,:,(2*generator.dim+1):-1].reshape(-1,generator.dim)
        dt = data[:,:,-1].reshape(-1)
        
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
        
    lr_schedule = learning_rate_fn()
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_schedule,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([generator.t_samples,
                                                                   generator.N_sim,
                                                                   3*generator.dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
    
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(s1s2_model) == hk.Transformed:
        if state is None:
            initial_params = s1s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
            if s1_params is not None:
                initial_params.update(s1_params)
            if s2_params is not None:
                initial_params.update(s2_params)
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s1s2_model.apply(params, rng_key, data)
    elif type(s1s2_model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = s1s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
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
                                                           output_shapes=([generator.t_samples,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
            train_dataset = iter(tfds.as_numpy(train_dataset))
            continue
        new_state, loss_val = update(state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in new_state.params[list(new_state.params.keys())[0]].values())) \
                and (loss_val<1e12)):
            state = new_state
        else:
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.t_samples,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
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

#%% Train t

def train_t(M:object,
            model:object,
            generator:object,
            state:TrainingState = None,
            lr_rate:float = 0.001,
            epochs:int=100,
            warmup_epochs:int=1000,
            save_step:int=100,
            optimizer:object=None,
            save_path:str = "",
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
        
        st_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
        dst_model = lambda x,y,t: grad(lambda t0: st_model(x,y,t0))(t)
    
        x0 = data[:,:,:generator.dim]
        xt = data[:,:,generator.dim:(2*generator.dim)]
        t = data[:,:,2*generator.dim]
        
        x0_t1, x0_t2 = x0[0], x0[-1]
        xt_t1, xt_t2 = xt[0], xt[-1]
        t1, t2 = t[0], t[-1]

        loss_s1 = st_model(x0.reshape(-1,generator.dim), xt.reshape(-1,generator.dim), t.reshape(-1,1)).reshape(*t.shape)
        loss_t1 = st_model(x0_t1, xt_t1, t1.reshape(-1,1)).squeeze()
        loss_t2 = st_model(x0_t2, xt_t2, t2.reshape(-1,1)).squeeze()
        loss_dt = vmap(lambda x,y,t: dst_model(x,y,t))(x0.reshape(-1,generator.dim), 
                                                       xt.reshape(-1,generator.dim), 
                                                       t.reshape(-1,1)).reshape(*t.shape)
        
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
        
    lr_schedule = learning_rate_fn()
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_schedule,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([generator.dt_steps,
                                                                   generator.N_sim,
                                                                   3*generator.dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(model) == hk.Transformed:
        if state is None:
            initial_params = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
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
                                                                           3*generator.dim+2]))
            train_dataset = iter(tfds.as_numpy(train_dataset))
            continue
        new_state, loss_val = update(state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in new_state.params[list(new_state.params.keys())[0]].values())) \
                and (loss_val<1e12)):
            state = new_state
        else:
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.dt_steps,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
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

#%% Score Training
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_p(M:object,
            model:object,
            generator:object,
            state:TrainingState = None,
            lr_rate:float = 0.001,
            epochs:int=100,
            warmup_epochs:int=1000,
            save_step:int=100,
            optimizer:object=None,
            save_path:str = "",
            loss_type:str='dsmvr',
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
        
        def s1_model(x,y,t):
            
            if x.ndim == 1:
                return grad(lambda y0: apply_fn(params, jnp.hstack((x,y0,t)), rng_key, state_val))(y)
            else:
                return vmap(lambda x,y,t: grad(lambda y0: apply_fn(params, jnp.hstack((x,y0,t)), rng_key, state_val))(y))(x,y,t)
            
        x0 = data[:,:,:generator.dim].reshape(-1, generator.dim)
        xt = data[:,:,generator.dim:(2*generator.dim)].reshape(-1,generator.dim)
        t = data[:,:,2*generator.dim].reshape(-1)
        dW = data[:,:,(2*generator.dim+1):-1].reshape(-1,generator.dim)
        dt = data[:,:,-1].reshape(-1)
        
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
    elif loss_type == "ssm":
        loss_model = ssm_s1fun()
    elif loss_type == "ssmvr":
        loss_model = ssmvr_s1fun()
    else:
        raise Exception("Invalid loss type. You can choose: vsm, dsm, dsmvr")
        
        return
        
    lr_schedule = learning_rate_fn()
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_schedule,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                   output_shapes=([generator.t_samples,
                                                                   generator.N_sim,
                                                                   3*generator.dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(model) == hk.Transformed:
        if state is None:
            initial_params = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:,:(2*generator.dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        if ((jnp.isnan(jnp.sum(data)))):
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.t_samples,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
            train_dataset = iter(tfds.as_numpy(train_dataset))
            continue
        new_state, loss_val = update(state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in new_state.params[list(new_state.params.keys())[0]].values())) \
                and (loss_val<1e12)):
            state = new_state
        else:
            generator.x0s = generator.x0s_default
            train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.float32,
                                                           output_shapes=([generator.t_samples,
                                                                           generator.N_sim,
                                                                           3*generator.dim+2]))
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