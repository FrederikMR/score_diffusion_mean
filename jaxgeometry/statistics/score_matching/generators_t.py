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

#%% Local Coordinates

class LocalSampling(object):
    
    def __init__(self,
                 M:object,
                 x0:Tuple[Array, Array],
                 repeats:int=2**3,
                 x_samples:int=2**5,
                 N_sim:int=2**8,
                 max_T:float=1.0,
                 dt_steps:int=100,
                 )->None:
        
        self.M = M
        self.x_samples=x_samples
        self.N_sim = N_sim
        self.max_T = max_T
        self.dt_steps = dt_steps
        self.repeats = repeats
        self.max_T = max_T
        if x0[0].ndim == 1:
            self.x0s = tile(x0, repeats)
        else:
            self.x0s = x0
        self.x0s_default = tile(x0, repeats)
        self._dts = dts(T=self.max_T, n_steps=self.dt_steps)
        self.counter = 0
        
        Brownian_coords(M)
        (product, sde_product, chart_update_product) = product_sde(M, 
                                                                   M.sde_Brownian_coords, 
                                                                   M.chart_update_Brownian_coords)
        
        self.product = product
        
    def __str__(self)->str:
        
        return "Generating Samples for Brownian Motion on Manifolds in Local Coordinates"
    
    def sim_diffusion_mean(self, 
                           x0:Tuple[Array, Array],
                           N_sim:int
                           )->Tuple[Array, Array]:
        
        x0s = tile(x0, N_sim)
        dW = dWs(N_sim*self.M.dim,self._dts).reshape(-1,N_sim,self.M.dim)
        (ts,xss,chartss,*_) = self.product(x0s,self._dts,dW,jnp.repeat(1.,N_sim))
        
        return (xss[-1], chartss[-1])
        
    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:
            
            dW = dWs(self.N_sim*self.M.dim,self._dts).reshape(-1,self.N_sim,self.M.dim)
            (ts,xss,chartss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                                jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                          self._dts,dW,jnp.repeat(1.,self.N_sim))
            
            Fx0s = self.x0s[0]
            self.x0s = (xss[-1,::self.x_samples],chartss[-1,::self.x_samples])
            
            if jnp.isnan(jnp.sum(xss)):
                self.x0s = self.x0s_default
                
            x0s = jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.dt_steps,1,1))
            xt = xss
            t = jnp.tile(ts, (self.N_sim, 1)).T.reshape(self.dt_steps, self.N_sim, 1)
            dt = jnp.tile(self._dts, (self.N_sim, 1)).T.reshape(self.dt_steps, self.N_sim, 1)
            
            yield jnp.concatenate((x0s,
                                   xt,
                                   t,
                                   dW,
                                   dt,
                                   ), axis=-1)
            
    def update_coords(self, 
                      x:Array
                      )->Tuple[Array,Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (Fx,chart)
    
    def grad_TM(self,
                x:Array,
                v:Array,
                )->Array:
        
        return v
    
    def grad_local(self,
                   x:Array,
                   v:Array
                   )->Array:
        
        return v
    
    def hess_TM(self,
                x:Array,
                v:Array,
                h:Array
                )->Array:
        
        return h
    
    def hess_local(self,
                   x:Array,
                   v:Array,
                   h:Array
                   )->Array:
        
        return h
    
#%% Projection from Chart

class EmbeddedSampling(object):
    
    def __init__(self,
                 M:object,
                 x0:Tuple[Array, Array],
                 repeats:int=2**3,
                 x_samples:int=2**5,
                 t_samples:int=2**7,
                 N_sim:int=2**8,
                 max_T:float=1.0,
                 dt_steps:int=1000,
                 )->None:
        
        self.M = M
        self.x_samples=x_samples
        self.t_samples = t_samples
        self.N_sim = N_sim
        self.max_T = max_T
        self.dt_steps = dt_steps
        self.repeats = repeats
        self.x0s = tile(x0, repeats)
        self.x0s_default = tile(x0, repeats)
        self._dts = dts(T=self.max_T, n_steps=self.dt_steps)
        
        Brownian_coords(M)
        (product, sde_product, chart_update_product) = product_sde(M, 
                                                                   M.sde_Brownian_coords, 
                                                                   M.chart_update_Brownian_coords)
        
        self.product = product
        
    def __str__(self)->str:
        
        return "Generating Samples for Brownian Motion on Manifolds using embedded chart"
    
    def sim_diffusion_mean(self, 
                           x0:Tuple[Array, Array],
                           N_sim:int
                           )->Tuple[Array, Array]:
        
        x0s = tile(x0, N_sim)
        dW = dWs(N_sim*self.M.dim,self._dts).reshape(-1,N_sim,self.M.dim)
        (ts,xss,chartss,*_) = self.product(x0s,self._dts,dW,jnp.repeat(1.,N_sim))
        
        return (xss[-1], chartss[-1])

    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:
            dW = dWs(self.N_sim*self.M.dim,self._dts).reshape(-1,self.N_sim,self.M.dim)
            (ts,xss,chartss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                                jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                          self._dts,dW,jnp.repeat(1.,self.N_sim))

            Fx0s = vmap(lambda x,chart: self.M.F((x,chart)))(*self.x0s) #x0s[1]
            self.x0s = (xss[-1,::self.x_samples],chartss[-1,::self.x_samples])
            
            if jnp.isnan(jnp.sum(xss)):
                self.x0s = self.x0s_default

            x0s = jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.dt_steps,1,1))
            xt = vmap(lambda x1,c1: vmap(lambda x,chart: self.M.F((x,chart)))(x1,c1))\
                (xss,chartss)
            t = jnp.tile(ts, (self.N_sim, 1)).T.reshape(self.dt_steps, self.N_sim, 1)
            dt = jnp.tile(self._dts, (self.N_sim, 1)).T.reshape(self.dt_steps, self.N_sim, 1)
            
            inds = jnp.array(random.sample(range(self._dts.shape[0]), self.t_samples))
            ts = ts[inds]
            samples = xss[inds]
            charts = chartss[inds]
            yield jnp.concatenate((x0s,
                                   xt,
                                   t,
                                   dW,
                                   dt,
                                   ), axis=-1)

    def update_coords(self, 
                      Fx:Array
                      )->Tuple[Array, Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (self.M.invF((Fx,chart)),chart)

    def grad_TM(self, 
                x:Array,
                v:Array,
                )->Array:

        x = self.update_coords(x)

        Fx = self.M.F(x)
        JFx = self.M.JF(x)
        Q, _ = jnp.linalg.qr(JFx)
        
        return jnp.dot(jnp.dot(Q,Q.T), s1_model(x0,Fx,t))
    
    def grad_local(self,
                   x:Array,
                   v:Array,
                   )->Array:

        Jf = self.M.JF(x)

        return jnp.einsum('ij,i->j', Jf, v)

    def hess_TM(self,
                x:Array,
                v:Array,
                h:Array,
                )->Array:
        
        val1 = self.M.proj(x, h)
        val2 = v-self.M.proj(x, v)
        val3 = jacfwd(lambda x: self.M.proj(x, val2))(x)
        
        return val1+val3
    
    def hess_local(self,
                   x:Array,
                   v:Array,
                   h:Array,
                   )->Array:
        
        x = self.update_coords(x)
        
        val1 = self.M.JF(x)
        val2 = jacfwdx(lambda x1: self.M.JF(x1))(x)
        term1 = jnp.einsum('jl,li,jk->ik', h, val1, val1)
        term2 = jnp.einsum('j,jik->ik', v, val2)
        
        return term1+term2

#%% Sampling in Tangent Space

class TMSampling(object):
    
    def __init__(self,
                 M:object,
                 x0:Tuple[Array, Array],
                 dim:int,
                 Exp_map:Callable[[Tuple[Array,Array], Array], Array]=None,
                 repeats:int=2**3,
                 x_samples:int=2**5,
                 N_sim:int=2**8,
                 max_T:float=1.0,
                 dt_steps:int=1000,
                 )->None:
        
        if not hasattr(M, "invJF"):
            M.invJF = lambda x: jnp.eye(M.emb_dim)[:M.dim]
        
        self.M = M
        self.x_samples=x_samples
        self.N_sim = N_sim
        self.max_T = max_T
        self.dt_steps = dt_steps
        self.repeats = repeats
        self.x0s = tile(x0, repeats)
        self.x0s_default = tile(x0, repeats)
        self.dim = dim
        
        self._dts = dts(T=self.max_T, n_steps=self.dt_steps)
        
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
        
        self.product = product
        
    def __str__(self)->str:
        
        return "Generating Samples for Brownian Motion on Manifolds using Projection in R^n"
    
    def sim_diffusion_mean(self, 
                           x0:Tuple[Array, Array],
                           N_sim:int
                           )->Tuple[Array, Array]:
        
        x0s = tile(x0, N_sim)
        dW = dWs(N_sim*self.dim,self._dts).reshape(-1,N_sim,self.dim)
        (ts,xss,chartss,*_) = self.product(x0s,self._dts,dW,jnp.repeat(1.,N_sim))
        
        return (chartss[-1], xss[-1])
        
    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:
            dW = dWs(self.N_sim*self.dim,self._dts).reshape(-1,self.N_sim,self.dim)
            (ts,xss,chartss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                                jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                          self._dts,dW,jnp.repeat(1.,self.N_sim))
            
            Fx0s = self.x0s[0]
            self.x0s = (xss[-1,::self.x_samples],chartss[-1,::self.x_samples])
            
            if jnp.isnan(jnp.sum(xss)):
                self.x0s = self.x0s_default
                
                
            x0s = jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.dt_steps,1,1))
            xt = xss
            t = jnp.tile(ts, (self.N_sim, 1)).T.reshape(self.dt_steps, self.N_sim, 1)
            dt = jnp.tile(self._dts, (self.N_sim, 1)).T.reshape(self.dt_steps, self.N_sim, 1)
            
            yield jnp.concatenate((x0s,
                                   xt,
                                   t,
                                   dW,
                                   dt,
                                   ), axis=-1)
            
    def update_coords(self, Fx:Array)->Tuple[Array,Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (self.M.invF((Fx,chart)), Fx)
    
    def grad_TM(self,
                x:Array,
                v:Array
                )->Array:
        
        return self.M.proj(x, v)
    
    def grad_local(self,
                   x:Array,
                   v:Array,
                   )->Array:
        
        x = self.update_coords(x)

        Jf = self.M.JF(x)

        return jnp.einsum('ij,i->j', Jf, v)

    def hess_TM(self,
                x:Array,
                v:Array,
                h:Array,
                )->Array:
        
        val1 = self.M.proj(x, h)
        val2 = v-self.M.proj(x, v)
        val3 = jacfwd(lambda x: self.M.proj(x, val2))(x)
        
        return val1+val3
    
    def hess_local(self,
                   x:Array,
                   v:Array,
                   h:Array,
                   )->Array:
        
        x = self.update_coords(x)
        
        val1 = self.M.JF(x)
        val2 = jacfwdx(lambda x1: self.M.JF(x1))(x)
        term1 = jnp.einsum('jl,li,jk->ik', h, val1, val1)
        term2 = jnp.einsum('j,jik->ik', v, val2)
        
        return term1+term2

#%% Sampling in using Projection

class ProjectionSampling(object):
    
    def __init__(self,
                 M:object,
                 x0:Tuple[Array, Array],
                 dim:int,
                 repeats:int=2**3,
                 x_samples:int=2**5,
                 t_samples:int=2**7,
                 N_sim:int=2**8,
                 max_T:float=1.0,
                 dt_steps:int=1000,
                 T_sample:bool = False,
                 t:float=.1,
                 reverse=True,
                 )->None:
        
        if not hasattr(M, "invJF"):
            M.invJF = lambda x: jnp.eye(M.emb_dim)[:M.dim]
        
        self.M = M
        self.x_samples=x_samples
        self.t_samples = t_samples
        self.N_sim = N_sim
        self.max_T = max_T
        self.dt_steps = dt_steps
        self.T_sample = T_sample
        self.t = t
        self.repeats = repeats
        self.x0s = tile(x0, repeats)
        self.x0s_default = tile(x0, repeats)
        self.dim = dim
        self.reverse=reverse
        
        self._dts = dts(T=self.max_T, n_steps=self.dt_steps)
        
        brownian_projection(M)
        (product,sde_product,chart_update_product) = product_sde(M, 
                                                                 M.sde_brownian_projection, 
                                                                 M.chart_update_brownian_projection, 
                                                                 integrator_stratonovich)
        
        self.product = product
        
    def __str__(self)->str:
        
        return "Generating Samples for Brownian Motion on Manifolds using Projection in R^n"
    
    def sim_diffusion_mean(self, 
                           x0:Tuple[Array, Array],
                           N_sim:int
                           )->Tuple[Array, Array]:
        
        x0s = tile(x0, N_sim)
        dW = dWs(N_sim*self.dim,self._dts).reshape(-1,N_sim,self.dim)
        (ts,xss,chartss,*_) = self.product(x0s,self._dts,dW,jnp.repeat(1.,N_sim))
        
        return (chartss[-1], xss[-1])
        
    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:
            dW = dWs(self.N_sim*self.dim,self._dts).reshape(-1,self.N_sim,self.dim)
            (ts,chartss,xss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                                jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                          self._dts,dW,jnp.repeat(1.,self.N_sim))
            
            if self.reverse:
                Fx0s = vmap(lambda x,chart: self.M.F((x,chart)))(self.x0s[1], self.x0s[0]) #x0s[1]
                self.x0s = (chartss[-1,::self.x_samples], xss[-1,::self.x_samples])
            else:
                Fx0s = vmap(lambda x,chart: self.M.F((x,chart)))(*self.x0s) #x0s[1]
                self.x0s = (xss[-1,::self.x_samples],chartss[-1,::self.x_samples])
                
            if jnp.isnan(jnp.sum(xss)):
                self.x0s = self.x0s_default
           
            if not self.T_sample:
                inds = jnp.array(random.sample(range(self._dts.shape[0]), self.t_samples))
                ts = ts[inds]
                samples = xss[inds]
                charts = chartss[inds]

                yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1)),
                                 vmap(lambda x,chart: self.M.F((x,chart)))(samples.reshape((-1,self.M.dim)),
                                                                      charts.reshape((-1,chartss.shape[-1]))), #charts.reshape(-1,chartss.shape[-1]), #
                                 jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                                 dW[inds].reshape(-1,self.dim),
                                 jnp.repeat(self._dts[inds],self.N_sim).reshape((-1,1)),
                                ))
            else:
                inds = jnp.argmin(jnp.abs(ts-self.t))
                ts = ts[inds]
                samples = xss[inds]
                charts = chartss[inds]
                yield jnp.hstack((jnp.repeat(Fx0s,self.x_samples,axis=0),
                                 vmap(lambda x,chart: self.M.F((x,chart)))(samples.reshape((-1,self.M.dim)),
                                                                           charts.reshape((-1,chartss.shape[-1]))), #charts.reshape(-1,chartss.shape[-1]), #
                                 jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                                 dW[inds].reshape(-1,self.dim),
                                 jnp.repeat(self._dts[inds],self.N_sim).reshape((-1,1)),
                                ))
            
    def update_coords(self, Fx:Array)->Tuple[Array,Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (self.M.invF((Fx,chart)), Fx)
    
    def grad_TM(self,
                x:Array,
                v:Array
                )->Array:
        
        return self.M.proj(x, v)
    
    def grad_local(self,
                   x:Array,
                   v:Array,
                   )->Array:

        Jf = self.M.JF(x)

        return jnp.einsum('ij,i->j', Jf, v)

    def hess_TM(self,
                x:Array,
                v:Array,
                h:Array,
                )->Array:
        
        val1 = self.M.proj(x, h)
        val2 = v-self.M.proj(x, v)
        val3 = jacfwd(lambda x: self.M.proj(x, val2))(x)
        
        return val1+val3
    
    def hess_local(self,
                   x:Array,
                   v:Array,
                   h:Array,
                   )->Array:
        
        x = self.update_coords(x)
        
        val1 = self.M.JF(x)
        val2 = jacfwdx(lambda x1: self.M.JF(x1))(x)
        term1 = jnp.einsum('jl,li,jk->ik', h, val1, val1)
        term2 = jnp.einsum('j,jik->ik', v, val2)
        
        return term1+term2
