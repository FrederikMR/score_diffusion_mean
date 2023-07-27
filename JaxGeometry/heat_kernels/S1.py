#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:44:36 2023

@author: fmry
"""

#%% Sources

#%% Modules

from src.setup import *
from src.params import *

#%% Code

def initialize(M:object, N_terms:int=20) -> None:
    
    @jit
    def get_coords(Fx:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

        chart = M.centered_chart(Fx)
        return (M.invF((Fx,chart)),chart)

    @jit
    def to_TM(Fx:jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        F = lambda x: jnp.array([jnp.cos(x[0]), jnp.sin(x[1])])
        JF = jacfwdx(F)
        
        x = get_coords(M, Fx)
        JFx = JF(x)
        
        return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

    @jit
    def to_TMchart(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:
        
        x = invF_spherical((Fx, Fx))
        JFx = M.Jf_spherical(x)
        
        return jnp.dot(JFx,v)

    @jit
    def to_TMx(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:

        x = get_coords(M, Fx)

        return jnp.dot(M.invJF((Fx,x[1])),v)
    
    @jit
    def hk(x:jnp.ndarray,y:jnp.ndarray,t:float)->float:
        
        def step(carry:float, k:int)->tuple[float, None]:
            
            carry += jnp.exp(-0.5*(2*jnp.pi*k+x1-y1)**2/t)
            
            return carry, None
        
        x1 = jnp.arctan2(x[1][1]/x[1][0]) % (2*jnp.pi)
        y1 = jnp.arctan2(y[1][1]/y[1][0]) % (2*jnp.pi)
       
        val, _ = lax.scan(step, init=jnp.zeros(1), xs=N_terms) 
       
        return val*const
    
    @jit
    def log_hk(x:jnp.ndarray,y:jnp.ndarray,t:float)->float:
        
        return jnp.log(hk(x,y,t))
    
    @jit
    def gradx_log_hk(x:jnp.ndarray,y:jnp.ndarray,t:float)->tuple[jnp.ndarray, jnp.ndarray]:
        
        def step(carry:float, k:int)->tuple[float, None]:
            
            term1 = 2*jnp.pi*k+x1-y1
            
            carry -= jnp.exp(-0.5*(term1**2)*tinv)*term1*tinv
            
            return carry, None
            
        x1 = jnp.arctan2(x[1][1]/x[1][0]) % (2*jnp.pi)
        y1 = jnp.arctan2(y[1][1]/y[1][0]) % (2*jnp.pi)
        tinv = 1/t
       
        val, _ = lax.scan(step, init=jnp.zeros(1), xs=N_terms) 
        grad = val*const/hk(x,y,t)
        
        grad_chart = to_TMchart(x[1], grad)
        grad_x = to_TMx(x[1], grad_chart)
       
        return grad_x, grad_chart
    
    @jit
    def grady_log_hk(x:jnp.ndarray, y:jnp.ndarray, t:float) -> tuple[jnp.ndarray, jnp.ndarray]:
        
        def step(carry:float, k:int)->tuple[float,None]:
            
            term1 = 2*jnp.pi*k+x1-y1
            
            carry += jnp.exp(-0.5*(term1**2)*tinv)*term1*tinv
            
            return carry, None
        
        x1 = jnp.arctan2(x[1][1]/x[1][0]) % (2*jnp.pi)
        y1 = jnp.arctan2(y[1][1]/y[1][0]) % (2*jnp.pi)
        tinv = 1/t
        
        val, _ = lax.scan(step, init=jnp.zeros(1), xs=N_terms) 
        grad = val*const/hk(x,y,t)
       
        grad_chart = to_TMchart(y[1], grad)
        grad_x = to_TMx(y[1], grad_chart)
       
        return grad_x, grad_chart
    
    @jit
    def gradt_log_hk(x:jnp.ndarray, y:jnp.ndarray, t:float)->float:
        
        def step1(carry:float, k:int)->tuple[float,None]:
            
            term1 = 0.5*(2*jnp.pi*k+x1-y1)**2
            
            carry += jnp.exp(term1/t)*term1/(t**2)
            
            return carry, None
        
        def step2(carry:float, k:int)->tuple[float,None]:
            
            carry += jnp.exp(-0.5*(2*jnp.pi*k+x1-y1)**2/t)
            
            return carry, None
        
        x1 = jnp.arctan2(x[1][1]/x[1][0]) % (2*jnp.pi)
        y1 = jnp.arctan2(y[1][1]/y[1][0]) % (2*jnp.pi)
            
        const2 = -1/(jnp.sqrt(jnp.pi)*(2*t)**(3/2))
       
        val1, _ = lax.scan(step1, init=jnp.zeros(1), xs=N_terms) 
        val1 *= const1
        
        val2, _ = lax.scan(step2, init=jnp.zeros(1), xs=N_terms) 
        val2 *= const2
       
        return (val1+val2)/hk(x,y,t)
    
    F_spherical = lambda x: jnp.array([jnp.cos(x[0]), jnp.sin(x[0])])
    invF_spherical = lambda x: jnp.arctan2(x[1][1]/x[1][0]) % (2*jnp.pi)
    Jf_spherical = jacfwdx(F_spherical)
    
    const = 1/jnp.sqrt(2*jnp.pi*t)
    
    M.hk = hk
    M.log_hk = log_hk
    M.gradx_log_hk = gradx_log_hk
    M.grady_log_hk = grady_log_hk
    M.gradt_log_hk = gradt_log_hk

    return