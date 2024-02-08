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
             loss_type:str='vsm',
             seed:int=2712
             )->None:
    
    def loss_vsm(params:hk.Params, state_val:dict, rng_key:Array, data:Array)->float:
        """ compute loss."""
        
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:2*N_dim]
        t = data[:,2*N_dim]
        #dW = data[:,(2*N_dim+1):-1]
        #dt = data[:,-1]
        
        s = apply_fn(params, data[:,:(2*N_dim+1)], rng_key, state_val)
        norm2s = jnp.sum(s*s, axis=1)
        
        s1 = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
        (xts, chartts) = vmap(generator.update_coords)(xt)
        
        divs = vmap(lambda x0, xt, chart, t: M.div((xt, chart), 
                                                   lambda x: generator.grad_local(s1, x0, x, t)))(x0,xts,chartts,t)
        
        return jnp.mean(norm2s+2.0*divs)

    def loss_dsm(params:hk.Params, state_val:dict, rng_key:Array, data:Array)->float:
        
        def f(x0,xt,t,dW,dt):
            
            #s1_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            #s1 = generator.grad_TM(s1_model, x0, xt, t)
            #s1_x0 = generator.grad_TM(s1_model, x0, x0, t)
            #dW = generator.dW_TM(xt,dW)
            
            #l1_loss = dW+dt*s1
            #l1_loss = jnp.sum(l1_loss*l1_loss)
            
            #eps = dt
            #z = -dW/jnp.sqrt(dt)
            
            #var_loss = eps*jnp.dot(z,z)-2*eps**(1.5)*jnp.dot(z, s1_x0)
            
            #return (l1_loss-var_loss)/(eps**2)
            
            s1 = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            s1 = generator.grad_TM(s1, x0, xt, t)
            dW = generator.dW_TM(xt,dW)

            loss = dW/dt+s1
            
            return jnp.mean(loss*loss)

        x0 = data[:,:N_dim]
        xt = data[:,N_dim:(2*N_dim)]
        t = data[:,2*N_dim]
        dW = data[:,(2*N_dim+1):-1]
        dt = data[:,-1]
        
        return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))
    
    @jit
    def update(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if loss_type == "vsm":
        loss_fun = jit(loss_vsm)
    elif loss_type == "dsm":
        loss_fun = jit(loss_dsm)
    else:
        print("Invalid loss function: Using Denoising Score Matching as default")
        loss_fun = jit(loss_dsm)
        
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
        if jnp.isnan(jnp.sum(data)):
            continue
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
             seed:int=2712
             )->None:
    
    @jit
    def loss_fun(params:hk.Params, 
                 state_val:dict, 
                 rng_key:Array, 
                 data:Array
                 )->float:
        
        #https://github.com/chenlin9/high_order_dsm/blob/main/functions/loss.py
        def f1(x0,xt,t,dW,dt):
            
            dW = generator.dW_TM(x0,dW)
            
            xp = x0+dW#dW
            xm = x0-dW#dW
            
            s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            
            s1 = generator.grad_TM(s1_model, x0, x0, t)
            s2 = generator.proj_hess(s1_model, s2_model, x0, x0, t)

            s1p = generator.grad_TM(s1_model, x0, xp, t)
            s2p = generator.proj_hess(s1_model, s2_model, x0, xp, t)
            
            s1m = generator.grad_TM(s1_model, x0, xm, t)
            s2m = generator.proj_hess(s1_model, s2_model, x0, xm, t)

            psi = s2+jnp.einsum('i,j->ij', s1, s1)
            psip = s2p+jnp.einsum('i,j->ij', s1p, s1p)
            psim = s2m+jnp.einsum('i,j->ij', s1m, s1m)
            diff = (jnp.eye(N_dim)-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
            
            loss1 = psip**2
            loss2 = psim**2
            loss3 = 2*diff*((psip-psi)+(psim-psi))
            
            loss_s2 = loss1+loss2+loss3
    
            return 0.5*jnp.sum(loss_s2)#jnp.mean(loss_s2)
        
        def f(x0,xt,t,dW,dt):
            
            dW = generator.dW_TM(xt,dW)
        
            s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            
            s1 = generator.grad_TM(s1_model, x0, xt, t)
            s2 = generator.proj_hess(s1_model, s2_model, x0, xt, t)
            #s1 = proj_grad(s1_model, x0, (xt,chart), t)
            #s2 = proj_hess(s1_model, s2_model, x0, (xt,chart), t)
            
            #loss_s2 = jnp.diag(s2)+s1*s1+(1-dW*dW/dt)/dt
            loss_s2 = s2+jnp.einsum('i,j->ij', s1, s1)+(jnp.eye(len(dW))-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
            
            return jnp.sum(loss_s2*loss_s2)
        
        def f2(x0,xt,t,dW,dt):
            
            xp = x0+generator.dW_embedded(x0,dW)#dW
            xm = x0-generator.dW_embedded(x0,dW)#dW
            
            dW = generator.dW_TM(x0,dW)
            
            s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            
            s1 = generator.grad_TM(s1_model, x0, x0, t)
            s2 = generator.proj_hess(s1_model, s2_model, x0, x0, t)

            s1p = generator.grad_TM(s1_model, x0, xp, t)
            s2p = generator.proj_hess(s1_model, s2_model, x0, xp, t)
            
            s1m = generator.grad_TM(s1_model, x0, xm, t)
            s2m = generator.proj_hess(s1_model, s2_model, x0, xm, t)

            psi = jnp.diag(s2)+s1*s1
            psip = jnp.diag(s2p)+s1p*s1p
            psim = jnp.diag(s2m)+s1m*s1m

            loss_s2 = psip**2+psim**2 \
                +2*((1-dW*dW/dt)/jnp.sqrt(dt))*\
                    (psip+psim-2*psi)
                                
            return jnp.mean(loss_s2)
        
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:(2*N_dim)]
        t = data[:,2*N_dim]
        noise = data[:,(2*N_dim+1):-1]
        dt = data[:,-1]
        
        loss = jnp.mean(vmap(
                        f,
                        (0,0,0,0,0))(x0,xt,t,noise,dt))
    
        return loss
    
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
        if jnp.isnan(jnp.sum(data)):
            continue
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

#%% Score Training for SN
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_s1s2(M:object,
               s1s2_model:object,
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
               seed:int=2712
               )->None:
    
    @jit
    def loss_fun(params:hk.Params, 
                 state_val:dict, 
                 rng_key:Array, 
                 data:Array
                 )->float:
        
        def s1_loss(x0,xt,t,dW,dt):
            
            s1 = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            s1 = generator.grad_TM(s1, x0, xt, t)
            dW = generator.dW_TM(xt,dW)

            loss = dW/dt+s1
            
            return jnp.sum(loss*loss)
        
        def s2_loss(x0,xt,t,dW,dt):
            
            dW = generator.dW_TM(xt,dW)
        
            s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            
            s1 = generator.grad_TM(s1_model, x0, xt, t)
            s2 = generator.proj_hess(s1_model, s2_model, x0, xt, t)
            #s1 = proj_grad(s1_model, x0, (xt,chart), t)
            #s2 = proj_hess(s1_model, s2_model, x0, (xt,chart), t)
            
            loss_s2 = jnp.diag(s2)+s1*s1+(1-dW*dW/dt)/dt
                            
            return jnp.sum(loss_s2*loss_s2)
        
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:(2*N_dim)]
        t = data[:,2*N_dim]
        noise = data[:,(2*N_dim+1):-1]
        dt = data[:,-1]
        
        loss_s1 = vmap(s1_loss,(0,0,0,0,0))(x0,xt,t,noise,dt)

        loss_s2 = vmap(s2_loss,(0,0,0,0,0))(x0,xt,t,noise,dt)
    
        return jnp.mean(loss_s1+loss_s2)
    
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
                                                   output_shapes=([batch_size,2*N_dim+dW_dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
    
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(s1s2_model) == hk.Transformed:
        if state is None:
            initial_params = s1s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s1s2_model.apply(params, rng_key, data)
    elif type(s1s2_model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = s1s2_model.init(jrandom.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s1s2_model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        data = next(train_dataset)
        if jnp.isnan(jnp.sum(data)):
            continue
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