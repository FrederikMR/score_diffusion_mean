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

#%% Maximumu Likelihood Geodesic Regression

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
                 lr_euc:float=0.01,
                 min_t:float=1e-3,
                 max_t:float=1.0,
                 max_step:float=0.1,
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
        self.min_t = min_t
        self.max_t = max_t
        self.max_step = max_step
        
        return
    
    def gradp(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], sigma:Array, v:Array, X:Array):
        
        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,chart,exp: self.grady_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1],exp_val)
        val2 = vmap(lambda w: self.gradp_exp(mu, w))(w)

        val = jnp.einsum('...i,...ij->...j', val1, val2)
        
        return -jnp.mean(val, axis=0)
    
    def gradv(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], sigma:Array, v:Array, X:Array):

        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,chart,exp: self.grady_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1],exp_val)
        val2 = vmap(lambda w: self.gradv_exp(mu, w))(w)
        
        term1 = jnp.einsum('kij,k->kij', val2, X)
        term2 = jnp.einsum('...j,...jk->...k', val1, term1)
        
        return -jnp.mean(term2, axis=0)
    
    def gradt(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], sigma:Array, v:Array, X:Array):
        
        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,chart, exp: self.gradt_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1], exp_val)
        
        return -2*jnp.mean(val1, axis=0)
    
    def gradient_optimization(self,
                              X_obs:Tuple[Array, Array],
                              X:Array,
                              )->None:
        
        @jit
        def p_gradient(p:Tuple[Array, Array], sigma:Array, v:Array)->Array:
            
            w = jnp.einsum('d,N->Nd', v, X)
            exp_val = vmap(lambda w: self.Exp(p,w))(w)
            val1 = vmap(lambda x,chart,exp: self.grady_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1],exp_val)
            val2 = vmap(lambda w: self.gradp_exp(p, w))(w)

            val = jnp.einsum('...i,...ij->...j', val1, val2)
            
            return -jnp.mean(val, axis=0)
        
        @jit
        def v_gradient(p:Tuple[Array, Array], sigma:Array, v:Array)->Array:

            w = jnp.einsum('d,N->Nd', v, X)
            exp_val = vmap(lambda w: self.Exp(p,w))(w)
            val1 = vmap(lambda x,chart,exp: self.grady_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1],exp_val)
            val2 = vmap(lambda w: self.gradv_exp(p, w))(w)
            
            term1 = jnp.einsum('kij,k->kij', val2, X)
            term2 = jnp.einsum('...j,...jk->...k', val1, term1)
            
            return -jnp.mean(term2, axis=0)
        
        @jit
        def sigma_gradient(p:Tuple[Array, Array], sigma:Array, v:Array)->Array:
            
            w = jnp.einsum('d,N->Nd', v, X)
            exp_val = vmap(lambda w: self.Exp(p,w))(w)
            val1 = vmap(lambda x,chart, exp: self.gradt_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1], exp_val)
            
            return -2.*jnp.mean(val1, axis=0)
        
        @jit
        def update(carry:Tuple[Array, Array, Array], idx:int
                   )->Tuple[Tuple[Array, Array, Array],
                            Tuple[Array, Array, Array]]:
            
            p, v, sigma = carry
            
            grad_p = p_gradient(p, sigma, v)
            grad_v = v_gradient(p, sigma, v)
            grad_sigma = sigma_gradient(p, sigma, v)
            
            grad_sigma = jnp.clip(grad_sigma, -self.max_step/self.lr_euc, self.max_step/self.lr_euc)
            
            p = self.M.Exp(p, -self.lr_rm*grad_p)
            p = self.M.update_coords(p, self.M.centered_chart(p))
            
            v -= self.lr_euc*grad_v
            sigma -= self.lr_euc*grad_sigma
            sigma = jnp.clip(sigma, self.min_t, self.max_t)

            return ((p, v, sigma),)*2
        
        #val, _ = lax.scan(update, init=(self.p, self.v, self.sigma), xs=jnp.ones(self.max_iter))
        p, v, sigma = self.p, self.v, self.sigma
        for i in range(self.max_iter):
            print(f"Epoch {i+1}/{self.max_iter}")
            p,v,sigma = update((p,v,sigma), i)[0]
            self.p = p
            self.v = v
            self.sigma = sigma
        
        #self.p = val[0]
        #self.v = val[1]
        #self.sigma = val[2]
        
        return
    
    def fit(self, X_obs:Tuple[Array, Array], X:Array, v:Array, p:Tuple[Array, Array], 
            sigma:Array, method="JAX")->None:
            
        self.p = p
        self.sigma = sigma
        self.v = v
        
        self.gradient_optimization(X_obs, X)
        
        return
    
#%% Maximumu Likelihood Geodesic Regression

class MLGeodesicRegressionEmbedded(object):
    def __init__(self,
                 M:object,
                 grady_log:Callable,
                 gradt_log:Callable,
                 Exp:Callable,
                 gradp_exp:Callable,
                 gradv_exp:Callable,
                 proj:Callable,
                 max_iter:int=100,
                 lr_rm:float=0.01,
                 lr_euc:float=0.01,
                 min_t:float=1e-3,
                 max_t:float=1.0,
                 max_step:float=0.1,
                 )->None:
        
        self.M = M
        self.grady_log = grady_log
        self.gradt_log = gradt_log
        self.gradp_exp = gradp_exp
        self.gradv_exp = gradv_exp
        self.Exp = Exp
        self.proj = proj
            
        self.max_iter=max_iter
        self.lr_rm = lr_rm
        self.lr_euc = lr_euc
        self.min_t = min_t
        self.max_t = max_t
        self.max_step = max_step
        
        return
    
    def gradp(self, X_obs:Array, mu:Array, sigma:Array, v:Array, X:Array):
        
        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,exp: self.grady_log(x,exp,sigma**2))(X_obs,exp_val)
        val2 = vmap(lambda w: self.gradp_exp(mu, w))(w)

        val = jnp.einsum('...i,...ij->...j', val1, val2)
        
        return -jnp.mean(val, axis=0)
    
    def gradv(self, X_obs:Array, mu:Array, sigma:Array, v:Array, X:Array):

        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,exp: self.grady_log(x,exp,sigma**2))(X_obs,exp_val)
        val2 = vmap(lambda w: self.gradv_exp(mu, w))(w)
        
        term1 = jnp.einsum('kij,k->kij', val2, X)
        term2 = jnp.einsum('...j,...jk->...k', val1, term1)
        
        return -jnp.mean(term2, axis=0)
    
    def gradt(self, X_obs:Array, mu:Array, sigma:Array, v:Array, X:Array):
        
        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x, exp: self.gradt_log(x,exp,sigma**2))(X_obs, exp_val)
        
        return -2*jnp.mean(val1, axis=0)
    
    def gradient_optimization(self,
                              X_obs:Array,
                              X:Array,
                              )->None:
        
        @jit
        def p_gradient(p:Array, sigma:Array, v:Array)->Array:
            
            p0 = p
            p = p/jnp.linalg.norm(p)
            v = self.proj(p,v)
            w = jnp.einsum('d,N->Nd', v, X)
            exp_val = vmap(lambda w: self.Exp(p,w))(w)
            val1 = vmap(lambda x,exp: self.grady_log(x,exp,sigma**2))(X_obs,exp_val)
            val2 = vmap(lambda w: self.gradp_exp(p, w))(w)
            val3 = jacfwd(lambda p: p/jnp.linalg.norm(p))(p0)

            val = jnp.einsum('...i,...ij->...j', val1, val2)
            val = jnp.einsum('...j,...ji->...i', val, val3)
            
            return -jnp.mean(val, axis=0)
        
        @jit
        def v_gradient(p:Array, sigma:Array, v:Array)->Array:

            p = p/jnp.linalg.norm(p)
            v = self.proj(p,v)
            w = jnp.einsum('d,N->Nd', v, X)
            exp_val = vmap(lambda w: self.Exp(p,w))(w)
            val1 = vmap(lambda x,exp: self.grady_log(x,exp,sigma**2))(X_obs,exp_val)
            val2 = vmap(lambda w: self.gradv_exp(p, w))(w)
            val3 = jacfwd(lambda v: self.proj(p,v))(v)
            
            term1 = jnp.einsum('ij,k->kij', val3, X)
            term2 = jnp.einsum('...ij,...jk->...ik', val2, term1)
            term3 = jnp.einsum('...j,...jk->...k', val1, term2)
            
            return -jnp.mean(term3, axis=0)
        
        @jit
        def sigma_gradient(p:Array, sigma:Array, v:Array)->Array:
            
            p = p/jnp.linalg.norm(p)
            v = self.proj(p,v)
            w = jnp.einsum('d,N->Nd', v, X)
            exp_val = vmap(lambda w: self.Exp(p,w))(w)
            val1 = vmap(lambda x, exp: self.gradt_log(x,exp,sigma**2))(X_obs, exp_val)
            
            return -2.*jnp.mean(val1, axis=0)
        
        @jit
        def update(carry:Tuple[Array, Array, Array], idx:int
                   )->Tuple[Tuple[Array, Array, Array],
                            Tuple[Array, Array, Array]]:
            
            p, v, sigma = carry
            
            grad_p = p_gradient(p, sigma, v)
            grad_v = v_gradient(p, sigma, v)
            grad_sigma = sigma_gradient(p, sigma, v)
            
            grad_sigma = jnp.clip(grad_sigma, -self.max_step/self.lr_euc, self.max_step/self.lr_euc)
            
            p = self.Exp(p, -self.lr_rm*grad_p)
            
            v -= self.lr_euc*grad_v
            sigma -= self.lr_euc*grad_sigma
            sigma = jnp.clip(sigma, self.min_t, self.max_t)

            return ((p, v, sigma),)*2
        
        #val, _ = lax.scan(update, init=(self.p, self.v, self.sigma), xs=jnp.ones(self.max_iter))
        p, v, sigma = self.p, self.v, self.sigma
        for i in range(self.max_iter):
            print(f"Epoch {i+1}/{self.max_iter}")
            p,v,sigma = update((p,v,sigma), i)[0]
            self.p = p
            self.v = v
            self.sigma = sigma
        
        #self.p = val[0]
        #self.v = val[1]
        #self.sigma = val[2]
        
        return
    
    def fit(self, X_obs:Array, X:Array, v:Array, p:Array, 
            sigma:Array, method="JAX")->None:
            
        self.p = p
        self.sigma = sigma
        self.v = v
        
        self.gradient_optimization(X_obs, X)
        
        return

#%%Old version

class MLGeodesicRegressionOld(object):
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
        
        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,chart,exp: self.grady_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1],exp_val)
        val2 = vmap(lambda w: self.gradp_exp(mu, w))(w)

        val = jnp.einsum('...i,...ij->...j', val1, val2)
        
        return -jnp.mean(val, axis=0)
    
    def gradv(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], sigma:Array, v:Array, X:Array):

        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,chart,exp: self.grady_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1],exp_val)
        val2 = vmap(lambda w: self.gradv_exp(mu, w))(w)
        
        term1 = jnp.einsum('kij,k->kij', val2, X)
        term2 = jnp.einsum('...j,...jk->...k', val1, term1)
        
        return -jnp.mean(term2, axis=0)
    
    def gradt(self, X_obs:Tuple[Array, Array], mu:Tuple[Array, Array], sigma:Array, v:Array, X:Array):
        
        w = jnp.einsum('d,N->Nd', v, X)
        exp_val = vmap(lambda w: self.Exp(mu,w))(w)
        val1 = vmap(lambda x,chart, exp: self.gradt_log((x,chart),exp,sigma**2))(X_obs[0], X_obs[1], exp_val)
        
        return -2*jnp.mean(val1, axis=0)
    
    def fit(self, X_obs:Tuple[Array, Array], X:Array, v:Array, mu:Tuple[Array, Array], 
            sigma:Array, method="JAX", opt="Joint")->None:
        
        gradv = lambda mu,y: self.gradv(X_obs, mu, y[0], y[1:], X)
        gradt = lambda mu,y: self.gradt(X_obs, mu, y[0], y[1:], X)
        grad_euc = lambda mu,y: jnp.concatenate((gradt(mu,y).reshape(-1), gradv(mu,y)))
        grad_rm = lambda mu,y: self.gradp(X_obs, mu, y[0], y[1:], X)
        
        if method == "JAX":
            x0_euc = jnp.concatenate((sigma, v))
            val, _ = JointJaxOpt(mu,x0_euc,self.M, grad_fn_rm = grad_rm,
                                  grad_fn_euc = grad_euc,max_iter=self.max_iter)
            mu, euc = val[0], val[1]
            mu = (mu[0], mu[1])
            sigma, v = euc[0], euc[1:]
        else:
            x0_euc = jnp.concatenate((sigma, v))
            val, _ = JointGradientDescent(mu,x0_euc,self. M,grad_fn_rm = grad_rm,
                                           grad_fn_euc = grad_euc,step_size_rm=self.lr_rm,
                                           step_size_euc=self.lr_euc,
                                           max_iter=self.max_iter)
            mu, euc = val[0], val[1]
            mu = (mu[0], mu[1])
            sigma, v = euc[0], euc[1:]
            
        self.mu = mu
        self.sigma = sigma
        self.v = v
        
        return