#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:43:13 2023

@author: fmry
"""

#%% Sources

#%% Modules

from src.setup import *
from src.params import *

#typing
from typing import Callable

#%% Code

def initialize(M:object, N_terms:int=20) -> None:
    
    @jit
    def get_coords(Fx:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

        chart = M.centered_chart(Fx)
        return (M.invF((Fx,chart)),chart)

    @jit
    def to_TM(Fx:jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        x = get_coords(M, Fx)
        JFx = M.JF(x)
        
        return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

    @jit
    def to_TMchart(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:
        
        x = get_coords(M, Fx)
        JFx = M.JF(x)
        return jnp.dot(JFx,v)

    @jit
    def to_TMx(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:

        x = get_coords(M, Fx)

        return jnp.dot(M.invJF((Fx,x[1])),v)
    
    @jit
    def hk(x:jnp.ndarray,y:jnp.ndarray,t:float) -> float:
        
        def sum_term(l:int, C_l:float) -> float:
        
            return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l
        
        def update_cl(l:int, Cl1:float, Cl2:float) -> float:
            
            return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
        
        def step(carry:tuple[float,float,float], l:int) -> tuple[tuple[float, float, float], None]:
    
            val, Cl1, Cl2 = carry
    
            C_l = update_cl(l, Cl1, Cl2)
    
            val += sum_term(l, C_l)
    
            return (val, C_l, Cl1), None
    
        x1 = x[1]
        y1 = y[1]
        xy_dot = jnp.dot(x1,y1)
        
        alpha = m1*0.5
        C_0 = 1.0
        C_1 = 2*alpha*xy_dot
        
        val = sum_term(0, C_0) + sum_term(1, C_1)
        
        grid = jnp.arange(2,N_terms,1)
        
        val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
            
        return val[0]*Am_inv/m1
    
    @jit
    def log_hk(x:jnp.ndarray,y:jnp.ndarray,t:float) -> float:
        
        return jnp.log(hk(x,y,t))
    
    @jit
    def gradx_log_hk(x:jnp.ndarray, y:jnp.ndarray, t:float) -> float:
        
        def sum_term(l:int, C_l1:float) -> float:
        
            return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l1
        
        def update_cl(l:int, Cl1:float, Cl2:float) -> float:
            
            return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
        
        def step(carry:tuple[float, float, float], l:int) -> tuple[tuple[float, float, float], None]:
            
            val, Cl1, Cl2 = carry
            
            C_l = update_cl(l-1, Cl1, Cl2)
            
            val += sum_term(l, C_l)
    
            return (val, C_l, Cl1), None

        x1 = x[1]
        y1 = y[1]
        
        xy_dot = jnp.dot(x1, y1)
        alpha = (M.dim+1)*0.5
        
        C_0 = 1.0
        C_1 = 2*alpha*xy_dot
        
        val = sum_term(1, C_0)+sum_term(2, C_1)
        
        grid = jnp.arange(3,N_terms,1)
        
        val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
        
        grad = val[0]*y1*Am_inv/hk(x,y,t)
        
        return (to_TMx(M, x1, grad), to_TM(M, x1, grad))
    
    @jit
    def grady_log_hk(x:jnp.ndarray, y:jnp.ndarray, t:float) -> float:
        
        def sum_term(l:int, C_l1:float) -> float:
        
            return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l1
        
        def update_cl(l:int, Cl1:float, Cl2:float) -> float:
            
            return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
        
        def step(carry:tuple[float, float, float], l:int)->tuple[tuple[float, float, float], None]:
            
            val, Cl1, Cl2 = carry
            
            C_l = update_cl(l-1, Cl1, Cl2)
            
            val += sum_term(l, C_l)
    
            return (val, C_l, Cl1), None

        x1 = x[1]
        y1 = y[1]
        
        xy_dot = jnp.dot(x1, y1)
        alpha = (M.dim+1)*0.5
        
        C_0 = 1.0
        C_1 = 2*alpha*xy_dot
        
        val = sum_term(1, C_0)+sum_term(2, C_1)
        
        grid = jnp.arange(3,N_terms,1)
        
        val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
        
        grad = val[0]*x1*Am_inv/hk(x,y,t)
        
        return (to_TMx(M, y1, grad), to_TM(M, y1, grad))
    
    @jit
    def gradt_log_hk(x:jnp.ndarray, y:jnp.ndarray, t:float) -> float:
        
        def sum_term(l:int, C_l:float) -> float:
        
            return -0.5*l*(l+m1)*jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l
        
        def update_cl(l:int, Cl1:float, Cl2:float) -> float:
            
            return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
        
        def step(carry:tuple[float, float, float], l:int) -> tuple[tuple[float, float, float], None]:
    
            val, Cl1, Cl2 = carry
    
            C_l = update_cl(l, Cl1, Cl2)
    
            val += sum_term(l, C_l)
    
            return (val, C_l, Cl1), None
        
        x1 = x[1]
        y1 = y[1]
        
        xy_dot = jnp.dot(x1,y1)
        
        alpha = m1*0.5
        C_0 = 1.0
        C_1 = 2*alpha*xy_dot
        
        val = sum_term(0, C_0) + sum_term(1, C_1)
        
        grid = jnp.arange(2,N_terms,1)
        
        val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
            
        return val[0]*Am_inv/(m1*hk(x,y,t))
    
    m1 = M.dim-1
    Am_inv = gamma((M.dim+1)*0.5)/(2*jnp.pi**((M.dim+1)*0.5))
    
    M.hk = hk
    M.log_hk = log_hk
    M.gradx_log_hk = gradx_log_hk
    M.grady_log_hk = grady_log_hk
    M.gradt_log_hk = gradt_log_hk

    return

#%% Gamma Function

def gamma(x:jnp.ndarray, T:float=100.0, N:int=1000, eps:float=1e-10) -> float:
    
    fun = lambda t: t**(x-1)*jnp.exp(-t)
    grid = jnp.linspace(eps+0.0, T, N)
    
    return simpson(fun, grid)

#%% Simpson rule

def simpson(func:Callable[[float], float], grid:jnp.ndarray, rule:str="3/8"):
   
    def simpson_13_step(carry, grid_vals):
       
        val, t0, f_prev = carry
        t, dt = grid_vals
       
        f_mid = func((t0+t)*0.5)
        f_up = func(t)
       
        val += (f_prev+4*f_mid+f_up)*dt

        return (val, t, f_up), None
   
    def simpson_38_step(carry, grid_vals):
       
        val, t0, f_prev = carry
        t, dt = grid_vals
       
        f_midprev = func((2*t0+t)/3)
        f_midup = func((t0+2*t)/3)
        f_up = func(t)
       
        val += (f_prev+3*f_midprev+3*f_midup+f_up)*dt

        return (val, t, f_up), None
   
    def simpson_composite_step(carry, grid_vals):
       
        val, f_prev = carry
        t0, t1, h0, h1 = grid_vals
        hph, hdh, hmh = h1 + h0, h1 / h0, h1 * h0
       
        f_up = func(t1)
        val += hph * ((2 - hdh) * f_prev + (hph**2 / hmh) * func(t0) + (2 - 1 / hdh) * f_up)

        return (val, f_up), None
   
    dt_grid = jnp.diff(grid)
    t0 = grid[0]
    new_grid = grid[1:]
   
    if rule == "1/3":
        yT, _ = lax.scan(simpson_13_step, (0.0, t0, func(t0)), xs=(new_grid, dt_grid))
        yT = yT[0]/6
    elif rule == "3/8":
        yT, _ = lax.scan(simpson_38_step, (0.0, t0, func(t0)), xs=(new_grid, dt_grid))
        yT = yT[0]*0.125
    elif rule == "Composite":
        N = len(grid)-1
       
        dt_grid_uneven, dt_grid_even, grid_uneven, grid_even = dt_grid[1:N:2], dt_grid[:(N-1):2], grid[1:N:2], grid[2:(N+1):2]
       
        yT, _ = lax.scan(simpson_composite_step, (0.0, func(t0)), xs=(grid_uneven, grid_even, dt_grid_even, dt_grid_uneven))
       
        yT = yT[0]
        if N%2 == 1:
            t_prev, t0, t1, h0, h1 = grid[N-2], grid[N-1], grid[N], dt_grid[N - 1], dt_grid[N]
            yT += func(t1)     * (2 * h1 ** 2 + 3 * h0 * h1) / (h0 + h1)
            yT += func(t0) * (h1 ** 2 + 3 * h1 * h0)     / h0
            yT -= func(t_prev) * h1 ** 3                     / (h0 * (h0 + h1))

        yT /= 6
   
    return yT


