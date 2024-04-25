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
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->float:
    
    def f(x0,xt,t,dW,dt):
        
        s1 = s1_model(x0, xt, t)
        #s1 = generator.grad_TM(x0, s1)
        dW = generator.grad_TM(x0, dW)

        loss = dW/dt+s1
        
        return jnp.sum(loss*loss)
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Variance Reduction Denoising Score Matching First Order

def dsmvr_s1fun(generator:object,
                s1_model,
                x0:Array,
                xt:Array,
                t:Array,
                dW:Array,
                dt:Array,
                )->float:
    
    def f(x0,xt,t,dW,dt):
        
        s1 = s1_model(x0,xt,t)
        s1p = s1_model(x0,x0,t)
        
        #s1 = generator.grad_TM(x0, s1)
        #s1p = generator.grad_TM(x0,s1p)
        dW = generator.grad_TM(x0, dW)
        
        l1_loss = dW/dt+s1
        l1_loss = 0.5*jnp.dot(l1_loss,l1_loss)
        var_loss = jnp.dot(s1p,dW)/dt+jnp.dot(dW,dW)/(dt**2)
        
        return l1_loss-var_loss
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

def dsm_s2fun(generator:object,
              s1_model,
              s2_model,
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->float:
    
    def f(x0,xt,t,dW,dt):
        
        s1 = s1_model(x0,xt,t)
        s2 = s2_model(x0,xt,t)
        
        s1 = generator.grad_TM(x0, s1)
        #s2 = generator.hess_TM(x0, s1, s2)
        dW = generator.grad_TM(x0, dW)

        loss_s2 = s2+jnp.einsum('i,j->ij', s1, s1)+(eye-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
        
        return jnp.sum(loss_s2*loss_s2)
    
    eye = jnp.eye(dW.shape[-1])
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Diag Second Order

def dsmdiag_s2fun(generator:object,
              s1_model,
              s2_model,
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->float:
    
    def f(x0,xt,t,dW,dt):
        
        s1 = s1_model(x0,xt,t)
        s2 = s2_model(x0,xt,t)
        
        s1 = generator.grad_TM(x0, s1)
        #s2 = generator.hess_TM(x0, s1, s2)
        dW = generator.grad_TM(x0, dW)

        loss_s2 = jnp.diag(s2)+s1*s1+(1.0-dW*dW/dt)/dt
        
        return jnp.sum(loss_s2*loss_s2)
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

def dsmvr_s2fun(generator:object,
                s1_model,
                s2_model,
                x0:Array,
                xt:Array,
                t:Array,
                dW:Array,
                dt:Array,
                )->float:
    
    def f(x0,xt,t,dW,dt):
                
        s1 = s1_model(x0,x0,t)
        s2 = s2_model(x0,x0,t)

        s1p = s1_model(x0,xt,t)
        s2p = s2_model(x0,xt,t)
        
        s1 = generator.grad_TM(x0, s1)
        #s2 = generator.hess_TM(x0, s1, s2)
        s1p = generator.grad_TM(x0, s1p)
        #s2p = generator.hess_TM(x0, s1p, s2p)
        dW = generator.grad_TM(x0, dW)

        psi = s2+jnp.einsum('i,j->ij', s1, s1)
        psip = s2p+jnp.einsum('i,j->ij', s1p, s1p)
        diff = (eye-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
        
        loss1 = psip**2
        loss2 = 2.*diff*(psip-psi)
        
        loss_s2 = loss1+loss2

        return 0.5*jnp.sum(loss_s2)
    
    eye = jnp.eye(dW.shape[-1])
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))

#%% Denoising Score Matching Second Order

def dsmdiagvr_s2fun(generator:object,
                    s1_model,
                    s2_model,
                    x0:Array,
                    xt:Array,
                    t:Array,
                    dW:Array,
                    dt:Array,
                    )->float:
    
    def f(x0,xt,t,dW,dt):
        
        dW = generator.dW_TM(x0,dW)
                
        s1 = s1_model(x0,x0,t)
        s2 = s2_model(x0,x0,t)

        s1p = s1_model(x0,xt,t)
        s2p = s2_model(x0,xt,t)
        
        s1 = generator.grad_TM(x0, s1)
        #s2 = generator.hess_TM(x0, s1, s2)
        s1p = generator.grad_TM(x0, s1p)
        #s2p = generator.hess_TM(x0, s1p, s2p)
        dW = generator.grad_TM(x0, dW)

        psi = jnp.diag(s2)+s1*s1
        psip = jnp.diag(s2p)+s1p*s1p
        diff = (1.0-dW*dW/dt)/dt
        
        loss1 = psip**2
        loss2 = 2.*diff*(psip-psi)
        
        loss_s2 = loss1+loss2

        return 0.5*jnp.sum(loss_s2)
    
    return jnp.mean(vmap(f,(0,0,0,0,0))(x0,xt,t,dW,dt))