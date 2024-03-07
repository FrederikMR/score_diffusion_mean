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
                 grady_log:Callable, 
                 gradt_log:Callable, 
                 n_clusters:int=4,
                 eps:float=0.01,
                 max_iter=100):
        
        self.grady_log = grady_log
        self.gradt_log = gradt_log
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.eps = eps
        self.key = jrandom.PRNGKey(2712)
        
        return
    
    def log_p(self):
        
        return
    
    def update_pi(self):
        
        return
    
    def update_theta(self):
        
        return