#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:33:37 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax
from jaxgeometry.setup import *

#jaxgeometry
from jaxgeometry.autodiff import jacfwdx
from jaxgeometry.integration import dts, dWs, integrator_stratonovich, integrator_ito
from jaxgeometry.stochastics import tile, product_sde, Brownian_coords, brownian_projection, GRW, \
    product_grw
    
#%% Riemannian Brownian Generator

class RiemannianBrownianGenerator(object):
    def __init__(self,
                 M:object,
                 x0:Tuple[Array,Array],
                 dim:int=None,
                 Exp_map:Callable[[Tuple[Array,Array], Array], Array]=None,
                 repeats:int=2**3,
                 x_samples:int=2**5,
                 t_samples:int=2**7,
                 T:float=1.0,
                 dt_steps:int=1000,
                 t0:float=0.0,
                 method:str='Local',
                 seed:int=2712,
                 )->None:
        
        if method not in ['Local', 'Embedded', 'TM', 'Projection']:
            raise ValueError(f"Passed method, {method}, is not: Local, Embedded, TM or Projection")
        else:
            self.method = method
            
        if dim is None:
            self.dim = M.dim
        else:
            self.dim = dim
            
        if method in ['Projection', 'TM']:
            x0 = (x0[1], x0[0])
        
        if x0[0].ndim == 1:
            x0s = tile(x0, repeats)
        else:
            x0s = x0
        self.x0s = x0s
        self.x0s_default = x0s
        
        self.M = M
        self.repeats = repeats
        self.x_samples = x_samples
        self.t_samples = t_samples
        self.N_sim = repeats*x_samples
        self.batch_size = repeats*x_samples*t_samples
        self.T = T
        self.dt_steps = dt_steps
        self.dt = jnp.array([T/dt_steps]*dt_steps)
        self.sqrtdt = jnp.sqrt(self.dt)
        self.t0 = t0
        self.key = jrandom.key(seed)
        self.counter = 0
        
        if self.method in ["Local", "Embedded"]:
            Brownian_coords(M)
            (product, sde_product, chart_update_product) = product_sde(M, 
                                                                       M.sde_Brownian_coords, 
                                                                       M.chart_update_Brownian_coords)
        elif self.method == "TM":
            if Exp_map is not None:
                GRW(M, f_fun = lambda x,v: M.ExpEmbedded(x[0], v))
                (product,sde_product,chart_update_product) = product_sde(M, 
                                                                         M.sde_grw, 
                                                                         M.chart_update_grw,
                                                                         lambda a,b: integrator_ito(a,b,lambda x,v: vmap(lambda x,y,v: M.ExpEmbedded(x,v))(x[0],x[1],v)))
            else:
                GRW(M, f_fun = lambda x,v: M.Exp(x, v))
                (product,sde_product,chart_update_product) = product_sde(M, 
                                                                         M.sde_grw, 
                                                                         M.chart_update_grw,
                                                                         lambda a,b: integrator_ito(a,b,lambda x,v: vmap(lambda x,y,v: M.Exp((x,y),v))(x[0],x[1],v)))
        elif self.method == "Projection":
            brownian_projection(M)
            (product,sde_product,chart_update_product) = product_sde(M, 
                                                                     M.sde_brownian_projection, 
                                                                     M.chart_update_brownian_projection, 
                                                                     integrator_stratonovich)
            
            
        self.product = product
        
        return
    
    def __str__(self)->str:
        
        return "Generator Object with Riemannian Brownian Motion paths"
    
    def update_coords(self, Fx:Array)->Tuple[Array,Array]:
        
        if self.method == "Local":
            chart = self.M.centered_chart(Fx)
            return (Fx,chart)
        else:
            return (self.M.invF((Fx,Fx)),Fx)
    
    def dWs(self, N_sim:int)->Array:
        
        keys = jrandom.split(self.key,num=2)
        self.key = keys[0]
        subkeys = keys[1:]
        
        normal = jrandom.normal(subkeys[0],(self.dt_steps,N_sim))
        
        return jnp.einsum('...,...j->...j', self.sqrtdt, normal)
    
    def grad_fun(self,
                 x0:Array,
                 xt:Tuple[Array,Array],
                 t:Array,
                 s1_model:Callable[[Array,Array,Array],Array],
                 )->Array:
        
        if self.method == "Local":
            return s1_model(x0,xt[0],t)
        else:
            Jf = self.M.JF(xt)
            Fx = self.M.F(xt)
            v = s1_model(x0,Fx,t)
            return jnp.einsum('ij,i->j', Jf, v)
        
    def grad_local(self,
                   x:Array,
                   v:Array
                   )->Array:
        
        if self.method == "Local":
            return v
        else:
            x = self.update_coords(x)
            Jf = self.M.JF((x[0],x[1]))
            return jnp.einsum('ij,i->j', Jf, v)
    
    def grad_TM(self,
                x:Array,
                v:Array
                )->Array:
        
        if self.method == "Local":
            return v
        else:
            return self.M.proj(x, v)
        
    def hess_local(self,
                   x:Array,
                   v:Array,
                   h:Array
                   )->Array:
        
        if self.method == "Local":
            return h
        else:
            x = self.update_coords(x)
            val1 = self.M.JF((x[0],x[1]))
            val2 = jacfwdx(self.M.JF)((x[0],x[1]))
            term1 = jnp.einsum('jl,li,jk->ik', h, val1, val1)
            term2 = jnp.einsum('j,jik', v, val2)
            return term1+term2

    def hess_TM(self,
                x:Array,
                v:Array,
                h:Array
                )->Array:
        
        if self.method == "Local":
            h
        else:
            val1 = self.M.proj(x, h)
            val2 = v-self.M.proj(x, v)
            val3 = jacfwd(lambda Fx: self.M.proj(Fx, val2))(x)
            return val1+val3
        
    def sim_diffusion_local(self, x0:Tuple[Array,Array], N_sim:int)->Tuple[Array, Array]:
        
        x0s = tile(x0, N_sim)
        dW = self.dWs(N_sim*self.M.dim).reshape(-1,N_sim,self.M.dim)
        (ts,xss,chartss,*_) = self.product(x0s,self.dt,dW,jnp.repeat(1.,N_sim))
        
        return (xss[-1], chartss[-1])
    
    def sim_diffusion_embedded(self, x0:Tuple[Array, Array], N_sim:int)->Tuple[Array, Array]:
        
        x0s = tile(x0, N_sim)
        dW = self.dWs(N_sim*self.M.dim).reshape(-1,N_sim,self.M.dim)
        (ts,xss,chartss,*_) = self.product(x0s,self.dt,dW,jnp.repeat(1.,N_sim))
        
        return (xss[-1], chartss[-1])
    
    def sim_diffusion_tm(self, x0:Tuple[Array, Array], N_sim:int)->Tuple[Array, Array]:
        
        x0s = tile(x0, N_sim)
        dW = self.dWs(N_sim*self.dim).reshape(-1,N_sim,self.dim)
        (ts,xss,chartss,*_) = self.product(x0s,self.dt,dW,jnp.repeat(1.,N_sim))
        
        return (chartss[-1], xss[-1])
    
    def sim_diffusion_projection(self, x0:Tuple[Array, Array], N_sim:int)->Tuple[Array, Array]:
        
        x0s = tile(x0, N_sim)
        dW = self.dWs(N_sim*self.dim).reshape(-1,N_sim,self.dim)
        (ts,xss,chartss,*_) = self.product(x0s,self.dt,dW,jnp.repeat(1.,N_sim))
        
        return (chartss[-1], xss[-1])

    def sim_diffusion_mean(self, x0:Tuple[Array, Array], N_sim:int)->Tuple[Array, Array]:
        
        if self.method in ['Projection', 'TM']:
            x0 = (x0[1], x0[0])
        
        if self.method == "Local":
            return self.sim_diffusion_local(x0,N_sim)
        elif self.method == "Embedded":
            return self.sim_diffusion_embedded(x0,N_sim)
        elif self.method == "TM":
            return self.sim_diffusion_tm(x0,N_sim)
        elif self.method == "Projection":
            return self.sim_diffusion_projection(x0,N_sim)
    
    def sample_local(self)->Array:
            
        dW = self.dWs(self.N_sim*self.M.dim).reshape(self.dt_steps,self.N_sim,self.M.dim)
        (ts,xss,chartss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                            jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                      self.dt,dW,jnp.repeat(1.,self.N_sim))
        
        Fx0s = self.x0s[0]
        self.x0s = (xss[-1,::self.x_samples],chartss[-1,::self.x_samples])
        
        if jnp.isnan(jnp.sum(xss)):
            self.x0s = self.x0s_default
        
        if self.t0 > 0:
            inds = jnp.argmin(jnp.abs(ts-self.t0))
            ts = ts[inds]
            samples = xss[inds]
            
            x0 = jnp.repeat(Fx0s,self.x_samples,axis=0)
            xt = samples.reshape(-1,self.M.dim)
            t = jnp.repeat(ts,self.N_sim).reshape((-1,1))
            dW = dW[inds].reshape(-1,self.M.dim)
            dt = jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1))
        else:
            inds = jnp.array(random.sample(range(self.dt_steps), self.t_samples))
            ts = ts[inds]
            samples = xss[inds]
            
            x0 = jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1))
            xt = samples.reshape(-1,self.M.dim)
            t = jnp.repeat(ts,self.N_sim).reshape((-1,1))
            dW = dW[inds].reshape(-1,self.M.dim)
            dt = jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1))
            
        return jnp.hstack((x0,xt,t,dW,dt))
    
    def sample_embedded(self)->Array:
        
        dW = self.dWs(self.N_sim*self.M.dim).reshape(-1,self.N_sim,self.M.dim)
        (ts,xss,chartss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                            jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                      self.dt,dW,jnp.repeat(1.,self.N_sim))

        Fx0s = vmap(lambda x,chart: self.M.F((x,chart)))(*self.x0s)
        self.x0s = (xss[-1,::self.x_samples],chartss[-1,::self.x_samples])
        
        if jnp.isnan(jnp.sum(xss)):
            self.x0s = self.x0s_default
       
        if self.t0 > 0.0:
            inds = jnp.argmin(jnp.abs(ts-self.t0))
            ts = ts[inds]
            samples = xss[inds]
            charts = chartss[inds]
            
            x0 = jnp.repeat(Fx0s,self.x_samples,axis=0)
            xt = vmap(lambda x,chart: self.M.F((x,chart)))(samples.reshape((-1,self.M.dim)),
                                                      charts.reshape((-1,chartss.shape[-1])))
            t = jnp.repeat(ts,self.N_sim).reshape((-1,1))
            dW = dW[inds].reshape(-1,self.M.dim)
            dt = jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1))
        else:
            inds = jnp.array(random.sample(range(self.dt_steps), self.t_samples))
            ts = ts[inds]
            samples = xss[inds]
            charts = chartss[inds]
            
            x0 = jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1))
            xt = vmap(lambda x,chart: self.M.F((x,chart)))(samples.reshape((-1,self.M.dim)),
                                                 charts.reshape((-1,chartss.shape[-1])))
            t = jnp.repeat(ts,self.N_sim).reshape((-1,1))
            dW = dW[inds].reshape(-1,self.M.dim)
            dt = jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1))
        
        return jnp.hstack((x0,xt,t,dW,dt))
    
    def sample_tm(self)->Array:
        
        dW = self.dWs(self.N_sim*self.dim).reshape(-1,self.N_sim,self.dim)
        (ts,xss,chartss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                            jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                      self.dt,dW,jnp.repeat(1.,self.N_sim))
        
        Fx0s = self.x0s[0]
        self.x0s = (xss[-1,::self.x_samples],chartss[-1,::self.x_samples])
        
        if jnp.isnan(jnp.sum(xss)):
            self.x0s = self.x0s_default
        
        if self.t0>0.0:
            inds = jnp.argmin(jnp.abs(ts-self.t0))
            ts = ts[inds]
            samples = xss[inds]
            
            x0 = jnp.repeat(Fx0s,self.x_samples,axis=0)
            xt = samples.reshape(-1,self.dim)
            t = jnp.repeat(ts,self.N_sim).reshape((-1,1))
            dW = dW[inds].reshape(-1,self.dim)
            dt = jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1))
        
        else:
            inds = jnp.array(random.sample(range(self.dt_steps), self.t_samples))
            ts = ts[inds]
            samples = xss[inds]
            
            x0 = jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1))
            xt = samples.reshape(-1,self.dim)
            t = jnp.repeat(ts,self.N_sim).reshape((-1,1))
            dW = dW[inds].reshape(-1,self.dim)
            dt = jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1))
            
        return jnp.hstack((x0,xt,t,dW,dt))
    
    def sample_projection(self)->Array:
        
        dW = self.dWs(self.N_sim*self.dim).reshape(-1,self.N_sim,self.dim)
        (ts,chartss,xss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                            jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                      self.dt,dW,jnp.repeat(1.,self.N_sim))

        Fx0s = vmap(lambda x,chart: self.M.F((x,chart)))(self.x0s[1], self.x0s[0]) #x0s[1]
        self.x0s = (chartss[-1,::self.x_samples], xss[-1,::self.x_samples])
            
        if jnp.isnan(jnp.sum(xss)):
            self.x0s = self.x0s_default
       
        if self.t0>0.0:
            inds = jnp.argmin(jnp.abs(ts-self.t0))
            ts = ts[inds]
            samples = xss[inds]
            charts = chartss[inds]
            
            x0 = jnp.repeat(Fx0s,self.x_samples,axis=0)
            xt = vmap(lambda x,chart: self.M.F((x,chart)))(samples.reshape((-1,self.M.dim)),
                                                      charts.reshape((-1,chartss.shape[-1])))
            dW = jnp.repeat(ts,self.N_sim).reshape((-1,1))
            dt = jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1))
        else:
            inds = jnp.array(random.sample(range(self.dt_steps), self.t_samples))
            ts = ts[inds]
            samples = xss[inds]
            charts = chartss[inds]
            
            x0 = jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1))
            xt = vmap(lambda x,chart: self.M.F((x,chart)))(samples.reshape((-1,self.M.dim)),
                                                 charts.reshape((-1,chartss.shape[-1])))
            t = jnp.repeat(ts,self.N_sim).reshape((-1,1))
            dW = dW[inds].reshape(-1,self.dim)
            dt = jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1))
            
        return jnp.hstack((x0,xt,t,dW,dt))
    
    def __call__(self)->Array:
        
        #self.counter += 1
        #if self.counter % 100 == 0:
        #    self.x0s = self.x0s_default
        while True:
            if self.method == "Local":
                yield self.sample_local()
            elif self.method == "Embedded":
                yield self.sample_embedded()
            elif self.method == "TM":
                yield self.sample_tm()
            elif self.method == "Projection":
                yield self.sample_projection()