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
from jaxgeometry.optimization.JAXOptimization import JointJaxOpt
from jaxgeometry.optimization.GradientDescent import JointGradientDescent

#%% Code

class BrownianMixture(object):
    def __init__(self, 
                 M:object,
                 grady_log:Callable, 
                 gradt_log:Callable, 
                 n_clusters:int=4,
                 eps:float=0.01,
                 method:str='Local',
                 update_method:str="Gradient",
                 iter_em:int=100,
                 iter_gradient:int=100,
                 lr_gradient:float=0.01,
                 dt_steps:int=100,
                 min_t:float=1e-3,
                 max_t:float=1.0,
                 seed:int=2712,
                 )->None:
        
        self.M = M
        self.grady_log = grady_log
        self.gradt_log = gradt_log
        self.n_clusters = n_clusters
        self.iter_em = iter_em
        self.iter_gradient = iter_gradient
        self.lr_gradient = lr_gradient
        self.eps = eps
        self.dt_steps = dt_steps
        self.update_method = update_method
        self.method = method
        self.min_t = min_t
        self.max_t = max_t
        self.key = jrandom.PRNGKey(seed)
        
    def __str__(self)->str:
        
        return "Riemannian Brownian Mixture Model Object"
    
    def p0(self, x_obs:Tuple[Array, Array], mu:Tuple[Array, Array], T:Array)->Array:
        
        if self.method == "Embedded":
            return jscipy.stats.multivariate_normal(x_obs[0],mu[0],self.eps*jnp.eye(mu[0]))
        else:
            return jscipy.stats.multivariate_normal(self.M.F(x_obs),self.M.F(mu),self.eps*jnp.eye(mu[1]))
    
    def density(self, 
                x_obs:Tuple[Array,Array], 
                mu:Tuple[Array, Array], 
                T:Array
                )->Array:
        
        def ode_step(carry, xs):
            
            log_qt, x, c = carry
            t,dt = xs
            
            x -= 0.5*self.M.div((x,c), lambda x: self.grady_log(mu, x, T-t))
            x += 0.5*self.M.div((x,c), lambda x: self.grady_log(mu, x, T-t))
            
            if self.M.do_chart_update is not None:
                update = self.M.do_chart_update(x)
                new_chart = self.M.centered_chart((x,c))
                new_x = self.M.update_coords((x,c),new_chart)[0]
                x, c = jnp.where(update,new_x,x),jnp.where(update,new_chart,c)
            
            return ((log_qt, x, c),)*2
        
        dt = jnp.linspace(0.0,T,self.dt_steps)
        t = jnp.cumsum(dt)
        _, val = lax.scan(ode_step, init=(0.0, *x_obs), xs=(t,dt))
        
        log_qt = val[0]
        
        return jnp.exp(log_qt)
    
    def gamma_znk(self, 
                  x_obs:Tuple[Array,Array], 
                  mu:Tuple[Array,Array], 
                  T:Array
                  )->Array:
        
        pt = self.density(x_obs,mu, T)
        p0 = self.p0(x_obs, mu, T)
        
        return p0*pt
    
    def update_pi(self, X_obs:Tuple[Array, Array])->Array:
        
        gamma_znk = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.gamma_znk((x,c),(mu_x,mu_c), t))(self.mu[0],
                                                                                                    self.mu[1],
                                                                                                    self.T))(X_obs[0],
                                                                                                             X_obs[1])
        val = jnp.einsum('ij,j->ij', gamma_znk, self.pi)
        sum_val = jnp.sum(val, axis=-1)
        
        return val/sum_val.reshape(-1,1)
    
    def update_theta(self, X_obs:Tuple[Array, Array])->Array:
        
        @jit
        def gradt_loss(y:Tuple[Array, Array],t:Array)->Array:
            
            gamma_zn = vmap(lambda x,c: self.gamma_znk((x,c), y, t))(X_obs[0],X_obs[1])
            
            s2 = vmap(lambda x,chart,gamma: gamma*self.gradt_log((x,chart),y,t))(X_obs[0], X_obs[1], gamma_zn)
            
            return -jnp.mean(s2, axis=0)
        
        @jit
        def gradx_loss(y:Tuple[Array,Array],t:Array)->Array:
            
            gamma_zn = vmap(lambda x,c: self.gamma_znk((x,c), y, t))(X_obs[0],X_obs[1])
            
            s1 = vmap(lambda x,chart,gamma: gamma*self.grady_log((x,chart),y,t))(X_obs[0], X_obs[1],gamma_zn)
            
            gradx = -jnp.mean(s1, axis=0)
            
            return gradx
        
        if self.update_method == "JAX":
            val = vmap(lambda mu_x,mu_c,t: JointJaxOpt((mu_x,mu_c),
                                                       t,
                                                       self.M,
                                                       grad_fn_rm = lambda y,t: gradx_loss(y, t),
                                                       grad_fn_euc = lambda y,t: gradt_loss(y, t),
                                                       max_iter=self.iter_gradient,
                                                       lr_rate = self.lr_gradient,
                                                       bnds_euc=(self.min_t,self.max_t),
                                                       )[0])(self.mu[0],self.mu[1],self.T)
            mu_sm, T_sm = val
        elif method == "Gradient":
            val = vmap(lambda mu_x,mu_c,t: JointGradientDescent((mu_x,mu_c),
                                                                t,
                                                                self.M,
                                                                grad_fn_rm = lambda y,t: gradx_loss(y, t),
                                                                grad_fn_euc = lambda y,t: gradt_loss(y, t),
                                                                step_size_rm=self.lr_gradient,
                                                                step_size_euc=self.lr_gradient,
                                                                max_iter = self.iter_gradient,
                                                                bnds_euc = (self.min_t,self.max_t),
                                                                )[0])(self.mu[0],self.mu[1],self.T)
            mu_sm, T_sm = val
        
        self.mu = (mu_sm[0], mu_sm[1])
        self.T = T_sm
        
        return
    
    def fit(self, 
            X_train:Tuple[Array,Array],
            mu_init:Tuple[Array,Array]=None,
            T_init:Array=None,
            pi_init:Array=None
            )->None:
        
        if pi_init is None:
            self.pi = jnp.array([1.0/self.n_clusters]*self.n_clusters)
        else:
            self.pi = pi_init
        
        if T_init is None:
            self.T = jnp.array([1.0]*self.n_clusters)
        else:
            self.T = T_init
        
        if mu_init is None:
            key, subkey = jrandom.split(self.key)
            self.key = subkey
            centroid_idx = [jrandom.choice(subkey, jnp.arange(0,len(X_train[0]), 1))]
            self.mu = (X_train[0][jnp.array(centroid_idx)].reshape(1,-1), 
                              X_train[1][jnp.array(centroid_idx)].reshape(1,-1))
        else:
            self.mu = mu_init
            
        for _ in range(self.iter_em):
            self.update_theta(X_train)
            self.update_pi(X_train)
        
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    