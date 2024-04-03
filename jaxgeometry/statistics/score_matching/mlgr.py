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
from jaxgeometry.optimization.JAXOptimization import JointJaxOpt, RMJaxOpt, JaxOpt
from jaxgeometry.optimization.GradientDescent import JointGradientDescent, RMGradientDescent, GradientDescent

#%% Code

class MLGeodesicRegression(object):
    def __init__(self,
                 M:object,
                 grady_log:Callable,
                 gradt_log:Callable,
                 Exp:Callable=None,
                 gradp_exp:Callable=None,
                 gradv_exp:Callable=None,
                 max_iter:int=100,
                 lr_rm:float=0.01,
                 lr_euc:float=0.01
                 )->None:
        
        self.M = M
        self.grady_log = grady_log
        self.gradt_log = gradt_log
        self.gradp_exp = gradp_exp
        self.gradv_exp = gradv_exp
        if Exp is None:
            self.Exp = M.Exp
        else:
            self.Exp = Exp
            
        self.max_iter=max_iter
        self.lr_rm = lr_rm
        self.lr_euc = lr_euc
        
        return
    
    def gradp(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], sigma:Array, v:Array, X:Array):
        
        v = v.reshape(-1, len(X))
        w = jnp.dot(v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,chart,exp: self.grady_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1],exp_val)
        val2 = vmap(lambda w: self.gradp_exp(mu, w))(w),

        val = jnp.einsum('...i,...ij->...j', val1, val2)
        
        return -jnp.mean(val, axis=0)
    
    def gradv(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], sigma:Array, v:Array, X:Array):
        
        v = v.reshape(-1, len(X))
        w = jnp.dot(v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = -vmap(lambda x,chart,exp: self.grady_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1],exp_val)
        val2 = vmap(lambda w: self.gradv_exp(mu, w))(w)
        
        term1 =jnp.einsum('kij,k->kij', val2, X)
        term2 = jnp.einsum('...j,...jk->...k', val1, term1)
        
        return -jnp.mean(term2, axis=0)
    
    def gradt(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], sigma:Array, v:Array, X:Array):
        
        v = v.reshape(-1, len(X))
        w = jnp.dot(v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = -vmap(lambda x,chart, exp: self.gradt_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1], exp_val)
        
        return -52*jnp.mean(val1, axis=0)
    
    def fit(self, X_obs:Tuple[Array, Array], X:Array, v:Array, mu:Tuple[Array, Array], 
            sigma:Array, method="JAX", opt="Joint")->None:
        
        gradv = lambda mu,y: self.gradv(X_obs, mu, y[0], y[1:], X)
        gradt = lambda mu,y: self.gradt(X_obs, mu, y[0], y[1:], X)
        grad_euc = lambda mu,y: jnp.concatenate((gradt(mu,y).reshape(-1), gradv(mu,y)))
        grad_rm = lambda mu,y: self.gradp(X_obs, mu, y[0], y[1:], X)
        
        if method == "JAX":
            x0_euc = jnp.concatenate((sigma, v))
            mu, euc, _, _ = JointJaxOpt(mu,x0_euc,self.M, grad_fn_rm = grad_rm,
                                        grad_fn_euc = grad_euc,max_iter=self.max_iter)
            mu = (mu[0][-1], mu[1][-1])
            sigma, v = euc[-1][0], euc[-1][1:]
        else:
            x0_euc = jnp.concatenate((sigma, v))
            mu, euc, _, _ = JointGradientDescent(mu,x0_euc,self. M,grad_fn_rm = grad_rm,
                                                 grad_fn_euc = grad_euc,step_size_rm=self.lr_rm,
                                                 step_size_euc=self.lr_euc,
                                                 max_iter=self.max_iter)
            mu = (mu[0][-1], mu[1][-1])
            sigma, v = euc[-1][0], euc[-1][1:]
            
        self.mu = mu
        self.sigma = sigma
        self.v = v
        
        return