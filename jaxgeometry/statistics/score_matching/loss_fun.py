#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources


#%% Modules

from jaxgeometry.setup import *

#%% Vanilla Score Matching First Order

@partial(jit, static_argnames=['generator', 's1_model'])
def vsm_s1(x0:Array,
           xt:Array,
           t:Array,
           dW:Array,
           dt:Array,
           generator:object,
           s1_model:Callable[[Array,Array,Array],Array],
           )->Array:
    """ compute loss."""
    
    loss_s1 = s1_model(x0,xt,t.reshape(-1,1))
    norm2s = jnp.sum(loss_s1*loss_s1, axis=1)
    
    (xts, chartts) = vmap(generator.update_coords)(xt)
    
    divs = vmap(lambda x0, xt, chart, t: generator.M.div((xt, chart), 
                                               lambda x: generator.grad_fun(x0, x, t, s1_model)))(x0,xts,chartts,t)
    
    return jnp.mean(norm2s+2.0*divs)

#%% Denoising Score Matching First Order

@partial(jit, static_argnames=['generator', 's1_model'])
def dsm_s1(x0:Array,
           xt:Array,
           t:Array,
           dW:Array,
           dt:Array,
           generator:object,
           s1_model:Callable[[Array,Array,Array],Array]
           )->Array:
    
    def f(x0,xt,t,dW,dt):
        
        s1 = s1_model(x0,xt,t)
        s1_loss = dW/dt+s1
        
        return jnp.sum(s1_loss*s1_loss)
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Variance Reduction Denoising Score Matching First Order

@partial(jit, static_argnames=['generator', 's1_model'])
def dsmvr_s1(x0:Array,
             xt:Array,
             t:Array,
             dW:Array,
             dt:Array,
             generator:object,
             s1_model:Callable[[Array,Array,Array],Array]
             )->Array:
    
    def f(x0,xt,t,dW,dt):
        
        dt_inv = 1/dt
        
        s1 = s1_model(x0,xt,t)
        s1p = s1_model(x0,x0,t)
        
        s1_loss = dW/dt+s1
        s1_loss = 0.5*jnp.dot(s1_loss,s1_loss)
        vr_loss = dt_inv*(jnp.dot(s1p,dW)+jnp.dot(dW,dW)*dt_inv)
        
        return s1_loss-vr_loss
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

@partial(jit, static_argnames=['generator', 's1_model', 's2_model'])
def dsm_s2(x0:Array,
           xt:Array,
           t:Array,
           dW:Array,
           dt:Array,
           generator:object,
           s1_model:Callable[[Array, Array, Array], Array],
           s2_model:Callable[[Array, Array, Array], Array],
           )->Array:
    
    def f(x0,xt,t,dW,dt):
        
        dt_inv = 1/dt
        
        s1 = lax.stop_gradient(s1_model(x0,xt,t))
        s2 = s2_model(x0,xt,t)

        s2_loss = s2+jnp.einsum('i,j->ij', s1, s1)+(eye-jnp.einsum('i,j->ij', dW, dW)*dt_inv)*dt_inv
        
        return jnp.sum(s2_loss*s2_loss)
    
    eye = jnp.eye(dW.shape[-1])
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Diag Second Order

@partial(jit, static_argnames=['generator', 's1_model', 's2_model'])
def dsmdiag_s2(x0:Array,
               xt:Array,
               t:Array,
               dW:Array,
               dt:Array,
               generator:object,
               s1_model:Callable[[Array, Array, Array], Array],
               s2_model:Callable[[Array, Array, Array], Array],
               )->Array:
    
    def f(x0,xt,t,dW,dt):
        
        dt_inv = 1/dt

        s1 = lax.stop_gradient(s1_model(x0,xt,t))
        s2 = s2_model(x0,xt,t)

        s2_loss = jnp.diag(s2)+s1*s1+(1.0-dW*dW*dt_inv)*dt_inv
        
        return jnp.sum(s2_loss*s2_loss)
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

@partial(jit, static_argnames=['generator', 's1_model', 's2_model'])
def dsmvr_s2(x0:Array,
             xt:Array,
             t:Array,
             dW:Array,
             dt:Array,
             generator:object,
             s1_model:Callable[[Array, Array, Array], Array],
             s2_model:Callable[[Array, Array, Array], Array],
             )->Array:
    
    def f(x0,xt,t,dW,dt):
        
        dt_inv = 1/dt
                
        s1 = lax.stop_gradient(s1_model(x0,x0,t))
        s2 = s2_model(x0,x0,t)

        s1p = lax.stop_gradient(s1_model(x0,xt,t))
        s2p = s2_model(x0,xt,t)

        psi = s2+jnp.einsum('i,j->ij', s1, s1)
        psip = s2p+jnp.einsum('i,j->ij', s1p, s1p)
        diff = (eye-jnp.einsum('i,j->ij', dW, dW)*dt_inv)*dt_inv
        
        loss1 = psip**2
        loss3 = 2.*diff*(psip-psi)
        
        s2_loss = loss1+loss3

        return 0.5*jnp.sum(s2_loss)
    
    eye = jnp.eye(dW.shape[-1])
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

@partial(jit, static_argnames=['generator', 's1_model', 's2_model'])
def dsmdiagvr_s2(x0:Array,
                 xt:Array,
                 t:Array,
                 dW:Array,
                 dt:Array,
                 generator:object,
                 s1_model:Callable[[Array, Array, Array], Array],
                 s2_model:Callable[[Array, Array, Array], Array],
                 )->Array:
    
    def f(x0,xt,t,dW,dt):
        
        dt_inv = 1/dt
                
        s1 = lax.stop_gradient(s1_model(x0,x0,t))
        s2 = s2_model(x0,x0,t)

        s1p = lax.stop_gradient(s1_model(x0,xt,t))
        s2p = s2_model(x0,xt,t)

        psi = jnp.diag(s2)+s1*s1
        psip = jnp.diag(s2p)+s1p*s1p
        diff = (1.0-dW*dW*dt_inv)*dt_inv
        
        loss1 = psip**2
        loss3 = 2.*diff*(psip-psi)
        
        s2_loss = loss1+loss3

        return 0.5*jnp.sum(s2_loss)
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))














