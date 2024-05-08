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

#%% Brownian Mixture Model

class BrownianMixture(object):
    def __init__(self, 
                 M:object,
                 log_hk:Callable,
                 grady_log:Callable, 
                 gradt_log:Callable, 
                 n_clusters:int=4,
                 eps:float=0.01,
                 method:str='Local',
                 update_method:str="Gradient",
                 max_iter:int=100,
                 lr:float=0.01,
                 dt_steps:int=100,
                 min_t:float=1e-3,
                 max_t:float=1.0,
                 seed:int=2712,
                 max_step:float=0.1,
                 )->None:
        
        self.M = M
        self.log_hk = log_hk
        self.grady_log = grady_log
        self.gradt_log = gradt_log
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.lr = lr
        self.eps = eps
        self.dt_steps = dt_steps
        self.update_method = update_method
        self.method = method
        self.min_t = min_t
        self.max_t = max_t
        self.key = jrandom.PRNGKey(seed)
        self.max_step = max_step
        
    def __str__(self)->str:
        
        return "Riemannian Brownian Mixture Model Object"
    
    def gamma_znk(self, 
                    X_obs:Tuple[Array,Array], 
                    mu:Tuple[Array, Array],
                    T:Array,
                    )->Array:
        
        log_pt = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.log_hk((x,c),(mu_x,mu_c), t))(self.mu[0],
                                                                                              self.mu[1],
                                                                                              self.T))(X_obs[0],
                                                                                                       X_obs[1])
        log_pt -= jnp.max(log_pt)
        pt = jnp.exp(log_pt)
        
        val = jnp.einsum('ij,j->ij', pt, self.pi)
        
        return val/jnp.sum(val, axis=-1).reshape(-1,1)
    
    def update_pi(self)->Array:
        
        return
    
    def update_mu(self)->Array:
        
        return
    
    def update_T(self)->Array:
        
        return
    
    def gradient_optimization(self,
                              X_obs:Tuple[Array, Array]
                              )->None:
        
        @jit
        def pi_gradient(pk:Array, pK:Array, pi:Array, mu:Tuple[Array, Array], T:Array)->Array:
            
            term1 = jnp.sum(jnp.einsum('ij,j->ij', pk, self.pi), axis=-1)
            term2 = (1-jnp.sum(self.pi))*pK
            denom = term1+term2
            
            num = pk-pK.reshape(-1,1)

            return -jnp.sum(num/denom.reshape(-1,1), axis=0)
        
        @jit
        def mu_gradient(pk:Array, pK:Array, pi:Array, mu:Tuple[Array, Array], T:Array)->Array:
            
            muk = (mu[0][:-1], mu[1][:-1])
            muK = (mu[0][-1], mu[1][-1])
            
            Tk = T[:-1]
            TK = T[-1]
            
            term1 = jnp.einsum('ij,j->ij', pk, self.pi)
            term2 = (1-jnp.sum(self.pi))*pK
            denom = (jnp.sum(term1, axis=-1)+term2).reshape(-1,1)
            
            s1k = vmap(lambda x,c: vmap(lambda mu_x, mu_c, t: self.grady_log((x,c), (mu_x, mu_c), t))(muk[0], 
                                                                                                      muk[1],
                                                                                                      Tk))(X_obs[0],
                                                                                                           X_obs[1])
            s1K = vmap(lambda x,c: self.grady_log((x,c), muK, TK))(X_obs[0],X_obs[1])

            grad_k = -jnp.sum(jnp.einsum('ij,ijk->ijk', term1/denom, s1k), axis=0)
            grad_K = -jnp.sum(jnp.einsum('i,ik->ik', term2/denom.reshape(-1), s1K), axis=0)
        
            return jnp.vstack((grad_k, grad_K))
        
        @jit
        def T_gradient(pk:Array, pK:Array, pi:Array, mu:Tuple[Array, Array], T:Array)->Array:
            
            muk = (mu[0][:-1], mu[1][:-1])
            muK = (mu[0][-1], mu[1][-1])
            
            Tk = T[:-1]
            TK = T[-1]
            
            term1 = jnp.einsum('ij,j->ij', pk, self.pi)
            term2 = (1-jnp.sum(self.pi))*pK
            denom = (jnp.sum(term1, axis=-1)+term2).reshape(-1,1)
            
            s1k = vmap(lambda x,c: vmap(lambda mu_x, mu_c, t: self.gradt_log((x,c), (mu_x, mu_c), t))(muk[0], 
                                                                                                      muk[1],
                                                                                                      Tk))(X_obs[0],
                                                                                                           X_obs[1])
            s1K = vmap(lambda x,c: self.gradt_log((x,c), muK, TK))(X_obs[0],X_obs[1])

            grad_k = -jnp.sum((term1/denom)*s1k, axis=0)
            grad_K = -jnp.sum(term2/denom.reshape(-1)*s1K, axis=0).reshape(1)
        
            return jnp.hstack((grad_k, grad_K))
        
        @jit
        def update(carry:Tuple[Array, Array, Array], idx:int
                   )->Tuple[Tuple[Array, Array, Array],
                            Tuple[Array, Array, Array]]:
            
            mu, T, pi = carry
            
            pk = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.log_hk((x,c),(mu_x,mu_c), t))(self.mu[0][:-1],
                                                                                              self.mu[1][:-1],
                                                                                              self.T[:-1]))(X_obs[0],
                                                                                                            X_obs[1])
            pK = vmap(lambda x,c: self.log_hk((x,c),(self.mu[0][-1],self.mu[1][-1]),self.T[-1]))(X_obs[0],X_obs[1])
            pK -= jnp.max(pk)
            pK = jnp.exp(pK)
            pk -= jnp.max(pk)
            pk = jnp.exp(pk)
                                                                                                        
            grad_pi = pi_gradient(pk, pK, pi, mu, T)
            grad_mu = mu_gradient(pk, pK, pi, mu, T)
            grad_T = T_gradient(pk, pK, pi, mu, T)
            grad_T = jnp.clip(grad_T, -jnp.ones_like(grad_T)*self.max_step/self.lr, 
                              jnp.ones_like(grad_T)*self.max_step/self.lr)
            
            pi -= self.lr*grad_pi
            pi = jnp.clip(pi, 0.0, 1.0)
            pi /= jnp.sum(pi)

            mu = vmap(lambda mu_x, mu_c, grad: self.M.Exp((mu_x, mu_c), -self.lr*grad))(mu[0], mu[1], grad_mu)
            mu = vmap(lambda mu_x, mu_c: self.M.update_coords((mu_x, mu_c),self.M.centered_chart((mu_x, mu_c))))(mu[0], mu[1])
            
            T -= self.lr*grad_T
            T = jnp.clip(T, self.min_t, self.max_t)

            return ((mu, T, pi),)*2
        
        pt = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.log_hk((x,c),(mu_x,mu_c), t))(self.mu[0],
                                                                                          self.mu[1],
                                                                                          self.T))(X_obs[0],
                                                                                                   X_obs[1])
        pt -= jnp.max(pt)
        pt = jnp.exp(pt)
        
        val, _ = lax.scan(update, init=(self.mu, self.T, self.pi), xs=jnp.ones(self.max_iter))
        
        self.mu = val[0]
        self.T = val[1]
        self.pi = val[2]
        
        return
    
    def fit(self, 
            X_train:Tuple[Array,Array],
            mu_init:Tuple[Array,Array]=None,
            T_init:Array=None,
            pi_init:Array=None
            )->None:
        
        if pi_init is None:
            self.pi = jnp.array([1.0/self.n_clusters]*self.n_clusters)[:-1]
        else:
            self.pi = pi_init
        
        if T_init is None:
            self.T = jnp.array([1.0/self.n_clusters]*self.n_clusters)
        else:
            self.T = T_init
        
        if mu_init is None:
            key, subkey = jrandom.split(self.key)
            self.key = subkey
            centroid_idx = jrandom.choice(subkey, jnp.arange(0,len(X_train[0]), 1), shape=(self.n_clusters,))
            self.mu = (X_train[0][jnp.array(centroid_idx)], 
                              X_train[1][jnp.array(centroid_idx)].reshape(len(centroid_idx),-1))
        else:
            self.mu = mu_init
            
        self.gradient_optimization(X_train)
        
        return

#%% Brownian Mixture Model

class BrownianMixtureGrad(object):
    def __init__(self, 
                 M:object,
                 log_hk:Callable,
                 grady_log:Callable, 
                 gradt_log:Callable, 
                 n_clusters:int=4,
                 eps:float=0.01,
                 method:str='Local',
                 update_method:str="Gradient",
                 max_iter:int=100,
                 lr:float=0.01,
                 dt_steps:int=100,
                 min_t:float=1e-3,
                 max_t:float=1.0,
                 seed:int=2712,
                 max_step:float=0.1,
                 )->None:
        
        self.M = M
        self.log_hk = log_hk
        self.grady_log = grady_log
        self.gradt_log = gradt_log
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.lr = lr
        self.eps = eps
        self.dt_steps = dt_steps
        self.update_method = update_method
        self.method = method
        self.min_t = min_t
        self.max_t = max_t
        self.key = jrandom.PRNGKey(seed)
        self.max_step = max_step
        
    def __str__(self)->str:
        
        return "Riemannian Brownian Mixture Model Object"
    
    def p0(self, x_obs:Tuple[Array, Array], mu:Tuple[Array, Array], T:Array)->Array:
        
        if self.method == "Local":
            return jscipy.stats.multivariate_normal.pdf(x_obs[0],mu[0],self.eps*jnp.eye(len(mu[0])))
        else:
            return jscipy.stats.multivariate_normal.pdf(self.M.F(x_obs),self.M.F(mu),self.eps*jnp.eye(len(mu[1])))
    
    def density(self, 
                x_obs:Tuple[Array,Array], 
                mu:Tuple[Array, Array], 
                T:Array
                )->Array:
        
        def ode_step(carry, xs):
            
            log_qt, x, c = carry
            t,dt = xs
            
            #x -= 0.5*self.M.div((x,c), lambda x: self.grady_log(mu, x, T-t))*dt
            log_qt -= 0.5*self.M.div((x,c), lambda x: self.grady_log(mu, x, T-t))*dt
            x += 0.5*self.grady_log(mu, (x,c), T-t)*dt
            
            if self.M.do_chart_update is not None:
                update = self.M.do_chart_update(x)
                new_chart = self.M.centered_chart((x,c))
                new_x = self.M.update_coords((x,c),new_chart)[0]
                x, c = jnp.where(update,new_x,x),jnp.where(update,new_chart,c)
            
            return ((log_qt, x, c),)*2
        
        t = jnp.linspace(0.0,T,self.dt_steps).reshape(-1)[:-1]
        dt = jnp.diff(t)
        val, carry = lax.scan(ode_step, init=(jnp.zeros(1, dtype=x_obs[0].dtype),x_obs[0].reshape(-1),
                                              x_obs[1].reshape(-1)), xs=(t[1:],dt))
        
        log_qt, x, c = val[0], val[1], val[2]
        p0 = self.p0((x,c), mu, T)
        pt = (p0*jnp.exp(log_qt)).squeeze()
        
        return jnp.exp(log_qt).squeeze()
    
    def p_nk(self, 
             X_obs:Tuple[Array,Array],
             mu:Tuple[Array, Array],
             T:Array,
             )->Array:
        
        pt = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.log_hk((x,c),(mu_x,mu_c), t))(self.mu[0],
                                                                                          self.mu[1],
                                                                                          self.T))(X_obs[0],
                                                                                                   X_obs[1])
        pt -= jnp.max(pt)
        pt = jnp.exp(pt)
        
        val = jnp.einsum('ij,j->ij', pt, self.pi)
        
        return val/jnp.sum(val, axis=-1).reshape(-1,1)
    
    def gamma_znk(self, 
                    X_obs:Tuple[Array,Array], 
                    mu:Tuple[Array, Array],
                    T:Array,
                    )->Array:
        
        pt = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.log_hk((x,c),(mu_x,mu_c), t))(self.mu[0],
                                                                                          self.mu[1],
                                                                                          self.T))(X_obs[0],
                                                                                                   X_obs[1])
        pt -= jnp.max(pt)
        pt = jnp.exp(pt)
        
        val = jnp.einsum('ij,j->ij', pt, self.pi)
        
        return val/jnp.sum(val, axis=-1).reshape(-1,1)
    
    def gradient_optimization(self,
                              X_obs:Tuple[Array, Array]
                              )->None:
        
        @jit
        def pi_gradient(pk:Array, pK:Array, pi:Array, mu:Tuple[Array, Array], T:Array)->Array:
            
            term1 = jnp.sum(jnp.einsum('ij,j->ij', pk, self.pi), axis=-1)
            term2 = (1-jnp.sum(self.pi))*pK
            denom = term1+term2
            
            num = pk-pK.reshape(-1,1)

            return -jnp.sum(num/denom.reshape(-1,1), axis=0)
        
        @jit
        def mu_gradient(pk:Array, pK:Array, pi:Array, mu:Tuple[Array, Array], T:Array)->Array:
            
            muk = (mu[0][:-1], mu[1][:-1])
            muK = (mu[0][-1], mu[1][-1])
            
            Tk = T[:-1]
            TK = T[-1]
            
            term1 = jnp.einsum('ij,j->ij', pk, self.pi)
            term2 = (1-jnp.sum(self.pi))*pK
            denom = (jnp.sum(term1, axis=-1)+term2).reshape(-1,1)
            
            s1k = vmap(lambda x,c: vmap(lambda mu_x, mu_c, t: self.grady_log((x,c), (mu_x, mu_c), t))(muk[0], 
                                                                                                      muk[1],
                                                                                                      Tk))(X_obs[0],
                                                                                                           X_obs[1])
            s1K = vmap(lambda x,c: self.grady_log((x,c), muK, TK))(X_obs[0],X_obs[1])

            grad_k = -jnp.sum(jnp.einsum('ij,ijk->ijk', term1/denom, s1k), axis=0)
            grad_K = -jnp.sum(jnp.einsum('i,ik->ik', term2/denom.reshape(-1), s1K), axis=0)
        
            return jnp.vstack((grad_k, grad_K))
        
        @jit
        def T_gradient(pk:Array, pK:Array, pi:Array, mu:Tuple[Array, Array], T:Array)->Array:
            
            muk = (mu[0][:-1], mu[1][:-1])
            muK = (mu[0][-1], mu[1][-1])
            
            Tk = T[:-1]
            TK = T[-1]
            
            term1 = jnp.einsum('ij,j->ij', pk, self.pi)
            term2 = (1-jnp.sum(self.pi))*pK
            denom = (jnp.sum(term1, axis=-1)+term2).reshape(-1,1)
            
            s1k = vmap(lambda x,c: vmap(lambda mu_x, mu_c, t: self.gradt_log((x,c), (mu_x, mu_c), t))(muk[0], 
                                                                                                      muk[1],
                                                                                                      Tk))(X_obs[0],
                                                                                                           X_obs[1])
            s1K = vmap(lambda x,c: self.gradt_log((x,c), muK, TK))(X_obs[0],X_obs[1])

            grad_k = -jnp.sum((term1/denom)*s1k, axis=0)
            grad_K = -jnp.sum(term2/denom.reshape(-1)*s1K, axis=0).reshape(1)
        
            return jnp.hstack((grad_k, grad_K))
        
        @jit
        def update(carry:Tuple[Array, Array, Array], idx:int
                   )->Tuple[Tuple[Array, Array, Array],
                            Tuple[Array, Array, Array]]:
            
            mu, T, pi = carry
            
            pk = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.log_hk((x,c),(mu_x,mu_c), t))(self.mu[0][:-1],
                                                                                              self.mu[1][:-1],
                                                                                              self.T[:-1]))(X_obs[0],
                                                                                                            X_obs[1])
            pK = vmap(lambda x,c: self.log_hk((x,c),(self.mu[0][-1],self.mu[1][-1]),self.T[-1]))(X_obs[0],X_obs[1])
            pK -= jnp.max(pk)
            pK = jnp.exp(pK)
            pk -= jnp.max(pk)
            pk = jnp.exp(pk)
                                                                                                        
            grad_pi = pi_gradient(pk, pK, pi, mu, T)
            grad_mu = mu_gradient(pk, pK, pi, mu, T)
            grad_T = T_gradient(pk, pK, pi, mu, T)
            grad_T = jnp.clip(grad_T, -jnp.ones_like(grad_T)*self.max_step/self.lr, 
                              jnp.ones_like(grad_T)*self.max_step/self.lr)
            
            pi -= self.lr*grad_pi
            pi = jnp.clip(pi, 0.0, 1.0)
            pi /= jnp.sum(pi)

            mu = vmap(lambda mu_x, mu_c, grad: self.M.Exp((mu_x, mu_c), -self.lr*grad))(mu[0], mu[1], grad_mu)
            mu = vmap(lambda mu_x, mu_c: self.M.update_coords((mu_x, mu_c),self.M.centered_chart((mu_x, mu_c))))(mu[0], mu[1])
            
            T -= self.lr*grad_T
            T = jnp.clip(T, self.min_t, self.max_t)

            return ((mu, T, pi),)*2
        
        pt = vmap(lambda x,c: vmap(lambda mu_x,mu_c,t: self.log_hk((x,c),(mu_x,mu_c), t))(self.mu[0],
                                                                                          self.mu[1],
                                                                                          self.T))(X_obs[0],
                                                                                                   X_obs[1])
        pt -= jnp.max(pt)
        pt = jnp.exp(pt)
        
        val, _ = lax.scan(update, init=(self.mu, self.T, self.pi), xs=jnp.ones(self.max_iter))
        
        self.mu = val[0]
        self.T = val[1]
        self.pi = val[2]
        
        return
    
    def fit(self, 
            X_train:Tuple[Array,Array],
            mu_init:Tuple[Array,Array]=None,
            T_init:Array=None,
            pi_init:Array=None
            )->None:
        
        if pi_init is None:
            self.pi = jnp.array([1.0/self.n_clusters]*self.n_clusters)[:-1]
        else:
            self.pi = pi_init
        
        if T_init is None:
            self.T = jnp.array([1.0/self.n_clusters]*self.n_clusters)
        else:
            self.T = T_init
        
        if mu_init is None:
            key, subkey = jrandom.split(self.key)
            self.key = subkey
            centroid_idx = jrandom.choice(subkey, jnp.arange(0,len(X_train[0]), 1), shape=(self.n_clusters,))
            self.mu = (X_train[0][jnp.array(centroid_idx)], 
                              X_train[1][jnp.array(centroid_idx)].reshape(len(centroid_idx),-1))
        else:
            self.mu = mu_init
            
        self.gradient_optimization(X_train)
        
        return

#%% Old Version

class BrownianMixtureOld(object):
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
        
        if self.method == "Local":
            return jscipy.stats.multivariate_normal.pdf(x_obs[0],mu[0],self.eps*jnp.eye(len(mu[0])))
        else:
            return jscipy.stats.multivariate_normal.pdf(self.M.F(x_obs),self.M.F(mu),self.eps*jnp.eye(len(mu[1])))
    
    def density(self, 
                x_obs:Tuple[Array,Array], 
                mu:Tuple[Array, Array], 
                T:Array
                )->Array:
        
        def ode_step(carry, xs):
            
            log_qt, x, c = carry
            t,dt = xs
            
            #x -= 0.5*self.M.div((x,c), lambda x: self.grady_log(mu, x, T-t))*dt
            log_qt -= 0.5*self.M.div((x,c), lambda x: self.grady_log(mu, x, T-t))*dt
            x += 0.5*self.grady_log(mu, (x,c), T-t)*dt
            
            if self.M.do_chart_update is not None:
                update = self.M.do_chart_update(x)
                new_chart = self.M.centered_chart((x,c))
                new_x = self.M.update_coords((x,c),new_chart)[0]
                x, c = jnp.where(update,new_x,x),jnp.where(update,new_chart,c)
            
            return ((log_qt, x, c),)*2
        
        t = jnp.linspace(0.0,T,self.dt_steps).reshape(-1)[:-1]
        dt = jnp.diff(t)
        val, carry = lax.scan(ode_step, init=(jnp.zeros(1, dtype=x_obs[0].dtype),x_obs[0].reshape(-1),
                                              x_obs[1].reshape(-1)), xs=(t[1:],dt))
        
        log_qt, x, c = val[0], val[1], val[2]
        p0 = self.p0((x,c), mu, T)
        pt = (p0*jnp.exp(log_qt)).squeeze()
        
        return pt
    
    def gamma_znk(self, 
                  X_obs:Tuple[Array,Array], 
                  )->Array:
        
        pt = vmap(lambda x, c: vmap(lambda mu_x, mu_c, t: self.log_hk((x, c), (mu_x, mu_c), t))(self.mu[0],
                                                                                                self.mu[1],
                                                                                                self.T))(X_obs[0],
                                                                                                         X_obs[1])
        pt -= jnp.max(pt)
        pt = jnp.exp(pt)

        val = jnp.einsum('ij,j->ij', pt, self.pi)

        return val/jnp.sum(val, axis=-1).reshape(-1, 1)

    def update_pi(self, X_obs:Tuple[Array, Array])->None:
        
        gamma_znk = self.gamma_znk(X_obs)
        self.pi = jnp.mean(gamma_znk, axis=0)
    
    def update_theta(self, X_obs:Tuple[Array, Array])->Array:
        
        @jit
        def gradt_loss(y:Tuple[Array, Array],t:Array)->Array:
            
            gamma_zn = jnp.sum(self.gamma_znk(X_obs), axis=-1)
            
            s2 = vmap(lambda x,chart,gamma: gamma*self.gradt_log((x,chart),y,t))(X_obs[0], X_obs[1], gamma_zn)
            
            return -jnp.mean(s2, axis=0)
        
        @jit
        def gradx_loss(y:Tuple[Array,Array],t:Array)->Array:
            
            gamma_zn = jnp.sum(self.gamma_znk(X_obs), axis=-1)
            
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
            mu_sm, T_sm = val[0], val[1]
        elif self.update_method == "Gradient":
            val = vmap(lambda mu_x,mu_c,t: JointGradientDescent((mu_x,mu_c),
                                                                t.reshape(-1),
                                                                self.M,
                                                                grad_fn_rm = lambda y,t: gradx_loss(y, t),
                                                                grad_fn_euc = lambda y,t: gradt_loss(y, t),
                                                                step_size_rm=self.lr_gradient,
                                                                step_size_euc=self.lr_gradient,
                                                                max_iter = self.iter_gradient,
                                                                bnds_euc = (self.min_t,self.max_t),
                                                                )[0])(self.mu[0],self.mu[1],self.T)
            mu_sm, T_sm = val[0], val[1]
        
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
            self.T = jnp.array([0.5]*self.n_clusters)
        else:
            self.T = T_init
        
        if mu_init is None:
            key, subkey = jrandom.split(self.key)
            self.key = subkey
            centroid_idx = jrandom.choice(subkey, jnp.arange(0,len(X_train[0]), 1), shape=(self.n_clusters,))
            self.mu = (X_train[0][jnp.array(centroid_idx)], 
                              X_train[1][jnp.array(centroid_idx)].reshape(len(centroid_idx),-1))
        else:
            self.mu = mu_init
            
        print(self.mu[0])
        for _ in range(self.iter_em):
            print(f"{_}")
            self.update_pi(X_train)
            self.update_theta(X_train)
        
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    