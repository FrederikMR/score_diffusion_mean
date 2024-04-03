#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:38:42 2023

@author: fmry
"""

#%% Sources

#https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670

#%% Modules

from jaxgeometry.setup import *

#%% Code

class BrownianMixture(object):
    def __init__(self, 
                 M:object,
                 grady_log:Callable, 
                 gradt_log:Callable, 
                 n_clusters:int=4,
                 eps:float=0.01,
                 max_iter:int=100
                 )->None:
        
        self.M = M
        self.grady_log = grady_log
        self.gradt_log = gradt_log
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.eps = eps
        self.key = jrandom.PRNGKey(2712)
        
        return
    
    def p0(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], t:Array)->Array:
        
        return
    
    def density(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], T:Array)->Array:
        
        def step_yt(carry, dt):
            
            t, x, c = carry
            
            t += dt
            
            x -= 0.5*M.div(self.grady_log(mu, (x,c), T-t))
            
            if self.M.do_chart_update is not None:
                update = self.M.do_chart_update(x)
                new_chart = self.M.centered_chart((x,c))
                new_x = self.M.update_coords((x,c),new_chart)[0]
                x, c = jnp.where(update,new_x,x),jnp.where(update,new_chart,c)
            
            return ((t,x,c),)*2
        
        def step_qt(carry, dt):
            
            t, x, c = carry
            
            t += dt
            
            x -= 0.5*M.div(self.grady_log(mu, (x,c), T-t))
            
            if self.M.do_chart_update is not None:
                update = self.M.do_chart_update(x)
                new_chart = self.M.centered_chart((x,c))
                new_x = self.M.update_coords((x,c),new_chart)[0]
                x, c = jnp.where(update,new_x,x),jnp.where(update,new_chart,c)
            
            return ((t,x,c),)*2
        
        lax.scan()
        
        step_yt()
        
        return
    
    def gamma_z(self, X_obs:Tuple[Array,Array], mu:Tuple[Array,Array], t:Array):
        
        log_p = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.log_p))
        
        
        return
    
    def log_p(self):
        
        return
    
    def update_pi(self):
        
        return
    
    def update_theta(self):
        
        return