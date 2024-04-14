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
    
    loss_s1 = s1_model(x0,xt,t)
    norm2s = jnp.sum(loss_s1*loss_s1, axis=1)
    
    (xts, chartts) = generator.update_coords(xt)
    
    divs = vmap(lambda x0, xt, chart, t: generator.M.div((xt, chart), 
                                               lambda x: generator.grad_local(s1_model, x0, x, t)))(x0,xts,chartts,t)
    
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
        
    dW = generator.grad_local(xt,dW)
    dt = dt.reshape(-1,1)
    
    s1 = grad_local(xt,s1_model(x0,xt,t))
    
    loss = dW/dt+s1
    
    return jnp.mean(jnp.sum(loss*loss, axis=-1))

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

    dW = generator.grad_local(xt,dW)
    dt = dt.reshape(-1,1)
    dt_inv = 1/dt
    
    s1 = generator.grad_local(xt,s1_model(x0,xt,t))
    s1p = generator.grad_local(xt,s1_model(x0,x0,t))
    
    dsm_loss = dW*dt_inv+s1
    
    s1_loss = 0.5*jnp.sum(dsm_loss*dsm_loss,axis=-1)
    vr_loss = jnp.sum(dt_inv*(s1p*dW+dt_inv*dW*dW), axis=-1)
    
    return jnp.mean(s1_loss-vr_loss)

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
    
    dW = generator.grad_local(xt,dW)
    dt_inv = 1/dt.reshape(-1,1,1)
    eye = jnp.eye(dW.shape[-1])
    
    v = s1_model(x0,xt,t)
    h = s2_model(x0,xt,t)
    s1 = generator.grad_local(xt,v)
    s2 = generator.hess_local(xt,v,h)
    
    diff = (eye-jnp.einsum('...i,...j->...ij', dW, dW)*dt_inv)*dt_inv
    loss = s2+jnp.einsum('...i,...j->...ij', s1, s1)+diff

    return jnp.mean(jnp.sum(loss*loss, axis=(1,2)))

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
    
    dW = generator.grad_local(xt,dW)
    dt = dt.reshape(-1,1)
    dt_inv = 1/dt
    
    v = s1_model(x0,xt,t)
    h = s2_model(x0,xt,t)
    h = vmap(lambda x: jnp.diag(x))(h)
    
    s1 = generator.grad_local(xt,v)
    s2 = jnp.diagonal(generator.hess_local(xt,v,h), axis1=1, axis2=2)
    
    loss = s2+s1*s1+(1.0-dW*dW*dt_inv)*dt_inv
    
    return jnp.mean(jnp.sum(loss*loss, axis=-1))

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
    
    dW = generator.grad_local(xt,dW)
    dt_inv = 1/dt.reshape(-1,1,1)
    eye = jnp.eye(dW.shape[-1])
    
    v = s1_model(x0,xt,t)
    vp = s1_model(x0,x0,t)
    h = s2_model(x0,xt,t)
    hp = s2_model(x0,x0,t)
    
    s1 = generator.grad_local(xt, v)
    s1p = generator.grad_local(xt, vp)
    s2 = generator.hess_local(xt, v, h)
    s2p = generator.hess_local(xt, vp, hp)

    psi = s2+jnp.einsum('...i,...j->...ij', s1, s1)
    psip = s2p+jnp.einsum('...i,...j->...ij', s1p, s1p)
    diff = (eye-jnp.einsum('...i,...j->...ij', dW, dW)*dt_inv)*dt_inv
    
    loss = 0.5*(psip**2)+diff*(psip-psi)

    return jnp.mean(jnp.sum(loss, axis=(1,2)))

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
    
    dW = generator.grad_local(xt,dW)
    dt = dt.reshape(-1,1)
    dt_inv = 1/dt
    
    v = s1_model(x0,xt,t)
    vp = s1_model(x0,x0,t)
    h = s2_model(x0,xt,t)
    h = vmap(lambda x: jnp.diag(x))(h)
    hp = s2_model(x0,x0,t)
    hp = vmap(lambda x: jnp.diag(x))(hp)
    
    s1 = generator.grad_local(xt, v)
    s1p = generator.grad_local(xt, vp)
    s2 = jnp.diagonal(generator.hess_local(xt, v, h), axis1=1, axis2=2)
    s2p = jnp.diagonal(generator.hess_local(xt, vp, hp), axis1=1,axis2=2)
    
    psi = s2+s1*s1
    psip = s2p+s1p*s1p
    diff = (1.0-dW*dW*dt_inv)*dt_inv
    
    loss = 0.5*(psip**2)+diff*(psip-psi)

    return jnp.mean(jnp.sum(loss, axis=-1))