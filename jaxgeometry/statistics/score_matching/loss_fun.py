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

def vsm_s1fun(generator:object,
              s1_model,
              params:hk.Params, 
              state_val:dict, 
              rng_key:Array, 
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array
              )->float:
    """ compute loss."""
    
    loss_s1 = s1_model(x0,xt,t.reshape(-1,1))
    norm2s = jnp.sum(loss_s1*loss_s1, axis=1)
    
    (xts, chartts) = vmap(generator.update_coords)(xt)
    
    divs = vmap(lambda x0, xt, chart, t: generator.M.div((xt, chart), 
                                               lambda x: generator.grad_local(s1_model, x0, x, t)))(x0,xts,chartts,t)
    
    return jnp.mean(norm2s+2.0*divs)

#%% Denoising Score Matching First Order

def dsm_s1fun(generator:object,
              s1_model,
              params:hk.Params, 
              state_val:dict, 
              rng_key:Array, 
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->float:
    
    def f(x0,xt,t,dW,dt):
        
        s1 = generator.grad_TM(s1_model, x0, xt, t)
        dW = generator.dW_TM(xt,dW)

        loss = dW/dt+s1
        
        return jnp.sum(loss*loss)
        
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
        
        #s1 = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
        #s1 = generator.grad_TM(s1, x0, xt, t)
        #dW = generator.dW_TM(xt,dW)

        #loss = dW/dt+s1
        
        #return jnp.mean(loss*loss)
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Variance Reduction Denoising Score Matching First Order

def dsmvr_s1fun(generator:object,
                s1_model,
                params:hk.Params, 
                state_val:dict, 
                rng_key:Array, 
                x0:Array,
                xt:Array,
                t:Array,
                dW:Array,
                dt:Array,
                )->float:
    
    def f(x0,xt,t,dW,dt):
        
        dW = generator.dW_TM(x0,dW)
        
        s1 = s1_model(x0,xt,t)
        s1p = s1_model(x0,x0,t)
        
        l1_loss = dW/dt+s1
        l1_loss = jnp.dot(l1_loss,l1_loss)
        var_loss = 2.*jnp.dot(s1p,dW)/dt+jnp.dot(dW,dW)/(dt**2)
        
        return l1_loss-var_loss
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

def dsm_s2fun(generator:object,
              s1_model,
              s2_model,
              params:hk.Params, 
              state_val:dict, 
              rng_key:Array, 
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->float:
    
    def f(x0,xt,t,dW,dt):
        
        dW = generator.dW_TM(xt,dW)    
        
        s1 = generator.grad_TM(s1_model, x0, xt, t)
        s2 = generator.proj_hess(s1_model, s2_model, x0, xt, t)

        loss_s2 = s2+jnp.einsum('i,j->ij', s1, s1)+(eye-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
        
        return jnp.sum(loss_s2*loss_s2)
    
    eye = jnp.eye(dW.shape[-1])
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Diag Second Order

def dsmdiag_s2fun(generator:object,
              s1_model,
              s2_model,
              params:hk.Params, 
              state_val:dict, 
              rng_key:Array, 
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->float:
    
    def f(x0,xt,t,dW,dt):
        
        dW = generator.dW_TM(xt,dW)    
        
        s1 = generator.grad_TM(s1_model, x0, xt, t)
        s2 = generator.proj_hess(s1_model, s2_model, x0, xt, t)

        loss_s2 = jnp.eye(s2)+s1*s1+(1-dW*dW/dt)/dt
        
        return jnp.sum(loss_s2*loss_s2)
    
    eye = jnp.eye(dW.shape[-1])
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

def dsmvr_s2fun(generator:object,
                s1_model,
                s2_model,
                params:hk.Params, 
                state_val:dict, 
                rng_key:Array, 
                x0:Array,
                xt:Array,
                t:Array,
                dW:Array,
                dt:Array,
                )->float:
    
    def f(x0,xt,t,dW,dt):
        
        dW = generator.dW_TM(x0,dW)
                
        s1 = generator.grad_TM(s1_model, x0, x0, t)
        s2 = s2_model(x0,x0,t)#generator.proj_hess(s1_model, s2_model, x0, x0, t)

        s1p = generator.grad_TM(s1_model, x0, xt, t)
        s2p = s2_model(x0,xt,t)#generator.proj_hess(s1_model, s2_model, x0, xt, t)
        
        #s1m = generator.grad_TM(s1_model, x0, xm, t)
        #s2m = generator.proj_hess(s1_model, s2_model, x0, xm, t)

        psi = s2+jnp.einsum('i,j->ij', s1, s1)
        psip = s2p+jnp.einsum('i,j->ij', s1p, s1p)
        #psim = s2m+jnp.einsum('i,j->ij', s1m, s1m)
        diff = (eye-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
        
        loss1 = psip**2
        #loss2 = psim**2
        loss3 = 2.*diff*(psip-psi)#2*diff*((psip-psi)+(psim-psi))
        
        loss_s2 = loss1+loss3#loss1+loss2+loss3

        return 0.5*jnp.sum(loss_s2)#jnp.mean(loss_s2)#jnp.mean(loss_s2)
    
    eye = jnp.eye(dW.shape[-1])
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

def dsmdiagvr_s2fun(generator:object,
                    s1_model,
                    s2_model,
                    params:hk.Params, 
                    state_val:dict, 
                    rng_key:Array, 
                    x0:Array,
                    xt:Array,
                    t:Array,
                    dW:Array,
                    dt:Array,
                    )->float:
    
    def f(x0,xt,t,dW,dt):
        
        dW = generator.dW_TM(x0,dW)
                
        s1 = generator.grad_TM(s1_model, x0, x0, t)
        s2 = s2_model(x0,x0,t)#generator.proj_hess(s1_model, s2_model, x0, x0, t)

        s1p = generator.grad_TM(s1_model, x0, xt, t)
        s2p = s2_model(x0,xt,t)#generator.proj_hess(s1_model, s2_model, x0, xt, t)

        psi = jnp.diag(s2)+s1*s1
        psip = jnp.diag(s2p)+s1p*s1p
        #psim = s2m+jnp.einsum('i,j->ij', s1m, s1m)
        diff = (1-dW*dW/dt)/dt
        
        loss1 = psip**2
        #loss2 = psim**2
        loss3 = 2.*diff*(psip-psi)#2*diff*((psip-psi)+(psim-psi))
        
        loss_s2 = loss1+loss3#loss1+loss2+loss3

        return 0.5*jnp.sum(loss_s2)#jnp.mean(loss_s2)#jnp.mean(loss_s2)
    
    eye = jnp.eye(dW.shape[-1])
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))