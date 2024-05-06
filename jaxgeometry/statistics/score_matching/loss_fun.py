#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources


#%% Modules

from jaxgeometry.integration import StdNormal
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
    
    divs = generator.div(x0,xt,t,s1_model)
    
    return jnp.mean(norm2s+2.0*divs)

#%% Sliced Score Matching First Order

class ssm_s1fun(object):
    def __init__(self,
                 M:int=1
                 ):
        self.M = M
        self.key = jrandom.PRNGKey(2712)
        
    def StdNormal(self, d:int, num:int)->Array:
        keys = jrandom.split(self.key,num=2)
        self.key = keys[0]
        subkeys = keys[1:]
        return jrandom.normal(subkeys[0],(num,d)).squeeze()
    
    def __call__(self,
                 generator:object,
                 s1_model,
                 x0:Array,
                 xt:Array,
                 t:Array,
                 dW:Array,
                 dt:Array,
                 )->float:
        
        #M = 1
        s1 = s1_model(x0,xt,t.reshape(-1,1))
        v = self.StdNormal(d=x0.shape[-1],num=self.M*x0.shape[0]).reshape(-1,x0.shape[-1])
        
        val = lambda x,y,t,v: grad(lambda y0: jnp.dot(v,s1_model(x,y0,t)))(y)
        
        vs1 = vmap(val)(x0,xt,t,v)
        J = 0.5*jnp.einsum('...j,...j->...',v,s1)**2
        
        return jnp.mean(J+jnp.einsum('...i,...i->...', vs1, v))

#%% Sliced Score Matching VR First Order

class ssmvr_s1fun(object):
    def __init__(self,
                 M:int=1
                 ):
        self.M = M
        self.key = jrandom.PRNGKey(2712)
        
    def StdNormal(self, d:int, num:int)->Array:
        keys = jrandom.split(self.key,num=2)
        self.key = keys[0]
        subkeys = keys[1:]
        return jrandom.normal(subkeys[0],(num,d)).squeeze()
    
    def __call__(self,
                 generator:object,
                 s1_model,
                 x0:Array,
                 xt:Array,
                 t:Array,
                 dW:Array,
                 dt:Array,
                 )->float:
        
        #M = 1
        s1 = s1_model(x0,xt,t.reshape(-1,1))
        v = self.StdNormal(d=x0.shape[-1],num=self.M*x0.shape[0]).reshape(-1,x0.shape[-1])
        
        val = lambda x,y,t,v: grad(lambda y0: jnp.dot(v,s1_model(x,y0,t)))(y)
        
        vs1 = vmap(val)(x0,xt,t,v)
        J = 0.5*jnp.einsum('...j,...j->...',s1,s1)
        
        return jnp.mean(J+jnp.einsum('...i,...i->...', vs1, v))

#%% Denoising Score Matching First Order

def dsm_s1fun(generator:object,
              s1_model,
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->float:
    
    def f(x0,xt,t,dW,dt,s1):

        s1 = generator.grad_TM(xt, s1)
        dW = generator.grad_TM(xt, dW)

        loss = dW/dt+s1
        
        return jnp.sum(loss*loss)
    
    s1 = s1_model(x0,xt,t.reshape(-1,1))
    
    return jnp.mean(vmap(f)(x0,xt,t,dW,dt,s1))

#%% Variance Reduction Denoising Score Matching First Order

def dsmvr_s1fun(generator:object,
                s1_model,
                x0:Array,
                xt:Array,
                t:Array,
                dW:Array,
                dt:Array,
                )->float:
    
    def f(x0,xt,t,dW,dt,s1,s1p):
        
        s1 = generator.grad_TM(xt, s1)
        s1p = generator.grad_TM(xt,s1p)
        dW = generator.grad_TM(xt, dW)
        
        l1_loss = dW/dt+s1
        l1_loss = 0.5*jnp.dot(l1_loss,l1_loss)
        var_loss = jnp.dot(s1p,dW)/dt+jnp.dot(dW,dW)/(dt**2)
        
        return l1_loss-var_loss
    
    s1 = s1_model(x0,xt,t.reshape(-1,1))
    s1p = s1_model(x0,x0,t.reshape(-1,1))
    
    return jnp.mean(vmap(f)(x0,xt,t,dW,dt,s1,s1p))

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
    
    def f(x0,xt,t,dW,dt,s1,s2):

        s1 = generator.grad_TM(xt, s1)
        s2 = generator.hess_TM(xt, s1, s2)
        dW = generator.grad_TM(xt, dW)

        loss_s2 = s2+jnp.einsum('i,j->ij', s1, s1)+(eye-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
        
        return jnp.sum(loss_s2*loss_s2)
    
    eye = jnp.eye(dW.shape[-1])
    s1 = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
    s2 = s2_model(x0,xt,t.reshape(-1,1))
    
    return jnp.mean(vmap(f)(x0,xt,t,dW,dt,s1,s2))

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
    
    def f(x0,xt,t,dW,dt,s1,s2):
        
        s1 = generator.grad_TM(xt, s1)
        s2 = generator.hess_TM(xt, s1, s2)
        dW = generator.grad_TM(xt, dW)

        loss_s2 = jnp.diag(s2)+s1*s1+(1.0-dW*dW/dt)/dt
        
        return jnp.sum(loss_s2*loss_s2)
    
    s1 = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
    s2 = s2_model(x0,xt,t.reshape(-1,1))
    
    return jnp.mean(vmap(f)(x0,xt,t,dW,dt,s1,s2))

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
    
    def f(x0,xt,t,dW,dt,s1,s1p,s2,s2p):
        
        s1 = generator.grad_TM(xt, s1)
        s2 = generator.hess_TM(xt, s1, s2)
        s1p = generator.grad_TM(xt, s1p)
        s2p = generator.hess_TM(xt, s1p, s2p)
        dW = generator.grad_TM(xt, dW)

        psi = s2+jnp.einsum('i,j->ij', s1, s1)
        psip = s2p+jnp.einsum('i,j->ij', s1p, s1p)
        diff = (eye-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
        
        loss1 = psip**2
        loss2 = 2.*diff*(psip-psi)
        
        loss_s2 = loss1+loss2

        return 0.5*jnp.sum(loss_s2)
    
    eye = jnp.eye(dW.shape[-1])
    s1 = lax.stop_gradient(s1_model(x0,x0,t.reshape(-1,1)))
    s1p = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
    s2 = s2_model(x0,x0,t.reshape(-1,1))
    s2p = s2_model(x0,xt,t.reshape(-1,1))
    
    return jnp.mean(vmap(f)(x0,xt,t,dW,dt,s1,s1p,s2,s2p))

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
    
    def f(x0,xt,t,dW,dt,s1,s1p,s2,s2p):
                
        s1 = lax.stop_gradient(s1_model(x0,x0,t))
        s2 = s2_model(x0,x0,t)

        s1p = lax.stop_gradient(s1_model(x0,xt,t))
        s2p = s2_model(x0,xt,t)
        
        s1 = generator.grad_TM(xt, s1)
        s2 = generator.hess_TM(xt, s1, s2)
        s1p = generator.grad_TM(xt, s1p)
        s2p = generator.hess_TM(xt, s1p, s2p)
        dW = generator.grad_TM(xt, dW)

        psi = jnp.diag(s2)+s1*s1
        psip = jnp.diag(s2p)+s1p*s1p
        diff = (1.0-dW*dW/dt)/dt
        
        loss1 = psip**2
        loss2 = 2.*diff*(psip-psi)
        
        loss_s2 = loss1+loss2

        return 0.5*jnp.sum(loss_s2)
    
    s1 = lax.stop_gradient(s1_model(x0,x0,t.reshape(-1,1)))
    s1p = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
    s2 = s2_model(x0,x0,t.reshape(-1,1))
    s2p = s2_model(x0,xt,t.reshape(-1,1))
    
    return jnp.mean(vmap(f)(x0,xt,t,dW,dt,s1,s1p,s2,s2p))