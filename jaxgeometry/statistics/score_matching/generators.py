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
                 t_samples:int=2**7,
                 N_sim:int=2**8,
                 max_T:float=1.0,
                 dt_steps:int=1000,
                 T_sample:bool = False,
                 t:float = 0.1
                 )->None:
        
        self.M = M
        self.x_samples=x_samples
        self.t_samples = t_samples
        self.N_sim = N_sim
        self.max_T = max_T
        self.dt_steps = dt_steps
        self.T_sample = T_sample
        self.t = t
        self.repeats = repeats
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
          #  self.counter += 1
          #  print(self.counter)
          #  if self.counter > 100:
          #      self.counter = 0
          #      self.x0s = self.x0s_default
            
            dW = dWs(self.N_sim*self.M.dim,self._dts).reshape(-1,self.N_sim,self.M.dim)
            (ts,xss,chartss,*_) = self.product((jnp.repeat(self.x0s[0],self.x_samples,axis=0),
                                                jnp.repeat(self.x0s[1],self.x_samples,axis=0)),
                                          self._dts,dW,jnp.repeat(1.,self.N_sim))
            
            Fx0s = self.x0s[0]
            self.x0s = (xss[-1,::self.x_samples],chartss[-1,::self.x_samples])
            
            if jnp.isnan(jnp.sum(xss)):
                self.x0s = self.x0s_default
            
            if not self.T_sample:
                inds = jnp.array(random.sample(range(self._dts.shape[0]), self.t_samples))
                ts = ts[inds]
                samples = xss[inds]
                
                yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1)),
                                  samples.reshape(-1,self.M.dim),
                                  jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                                  dW[inds].reshape(-1,self.M.dim),
                                  jnp.repeat(self._dts[inds],self.N_sim).reshape((-1,1)),
                                  ))
            
            else:
                inds = jnp.argmin(jnp.abs(ts-self.t))
                ts = ts[inds]
                samples = xss[inds]
                yield jnp.hstack((jnp.repeat(Fx0s,self.x_samples,axis=0),
                                  samples.reshape(-1,self.M.dim),
                                  jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                                  dW[inds].reshape(-1,self.M.dim),
                                  jnp.repeat(self._dts[inds],self.N_sim).reshape((-1,1)),
                                  ))
            
    def update_coords(self, Fx:Array)->Tuple[Array,Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (Fx,chart)
    
    def grad_TM(self,
                  s1_model:Callable[[Array, Array, Array], Array], 
                  x0:Array, 
                  x:Array, 
                  t:Array
                  )->Array:
        
        return s1_model(x0, x, t)
    
    def grad_local(self,
                   s1_model:Callable[[Array, Array, Array], Array], 
                   x0:Array, 
                   x:Tuple[Array,Array], 
                   t:Array
                   )->Array:
        
        return s1_model(x0, x[0], t)
    
    def proj_hess(self,s1_model:Callable[[Array, Array, Array], Array], 
                   s2_model:Callable[[Array, Array, Array], Array],
                   x0:Array, 
                   x:Array, 
                   t:Array
                   )->Array:
        
        return s2_model(x0,x,t)
    
    def dW_TM(self,
                x:Array,
                dW:Array
                )->Array:
    
        return dW
    
    def dW_local(self,
                x:Array,
                dW:Array
                )->Array:
    
        return dW
    
    def dW_embedded(self,
                x:Array,
                dW:Array
                )->Array:
        
        return dW

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
                 T_sample:bool = False,
                 t:float = 0.1
                 )->None:
        
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
           
            if not self.T_sample:
                inds = jnp.array(random.sample(range(self._dts.shape[0]), self.t_samples))
                ts = ts[inds]
                samples = xss[inds]
                charts = chartss[inds]
                yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1)),
                                 vmap(lambda x,chart: self.M.F((x,chart)))(samples.reshape((-1,self.M.dim)),
                                                                      charts.reshape((-1,chartss.shape[-1]))), #charts.reshape(-1,chartss.shape[-1]), #
                                 jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                                 dW[inds].reshape(-1,self.M.dim),
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
                                 dW[inds].reshape(-1,self.M.dim),
                                 jnp.repeat(self._dts[inds],self.N_sim).reshape((-1,1)),
                                ))

    def update_coords(self, Fx:Array)->Tuple[Array, Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (self.M.invF((Fx,chart)),chart)

    def grad_TM(self, 
                  s1_model:Callable[[Array, Array, Array], Array],
                  x0:Array, 
                  x:Array, 
                  t:Array):

        x = self.update_coords(x)

        Fx = self.M.F(x)
        JFx = self.M.JF(x)
        Q, _ = jnp.linalg.qr(JFx)
        
        return jnp.dot(jnp.dot(Q,Q.T), s1_model(x0,Fx,t))
    
    def grad_local(self, 
                  s1_model:Callable[[Array, Array, Array], Array],
                  x0:Array, 
                  x:Tuple[Array,Array], 
                  t:Array):

        Fx = self.M.F(x)
        invJFx = self.M.invJF((x[1],x[1]))
        
        return jnp.tensordot(invJFx,s1_model(x0,Fx,t),(1,0))

    def proj_hess(self,
                  s1_model:Callable[[Array, Array, Array], Array], 
                  s2_model:Callable[[Array, Array, Array], Array],
                  x0:Array, 
                  x:Array, 
                  t:Array
                  )->Array:
        
        #x = self.update_coords(x)
        
        #Fx = self.M.F(x)
        #JFx = self.M.JF(x)
        #Q, _ = jnp.linalg.qr(JFx)
        
        
        #return jnp.dot(jnp.dot(Q,Q.T), s2_model(x0,Fx,t))
        
        #return s2_model(x0,Fx,t)
        
        #Fx = self.M.F(x)
        #invJFx = self.M.invJF((x[1],x[1]))
        
        #return jnp.tensordot(invJFx,s2_model(x0,Fx,t),(1,0))
        
        x = self.update_coords(x)

        Fx = self.M.F(x)
        JFx = self.M.JF(x)        
        
        val1 = self.M.proj(x0, s2_model(x0,Fx,t))
        val2 = s1_model(x0,Fx,t)-self.M.proj(x0, s1_model(x0,Fx,t))
        val3 = jacfwd(lambda x: self.M.proj(x, val2))(x0)
        
        
        return val1+val3#jnp.einsum('i,j->ij', M.proj(x0, s1_model(x0,Fx,t)), x)
        
        #return jnp.dot(jnp.dot(Q,Q.T), s2_model(x0,Fx,t))
    
    def dW_TM(self,
              x:Array,
              dW:Array
              )->Array:
        
        x = self.update_coords(x)
        
        return jnp.dot(self.M.JF(x), dW)
    
    def dW_local(self,
                x:Array,
                dW:Array
                )->Array:
        
        return jnp.dot(self.M.invJF((x[1],x[1])), dW)
    
    def dW_embedded(self,
                x:Array,
                dW:Array
                )->Array:
        
        x = self.update_coords(x)
        
        JFx = self.M.JF(x)
        
        return jnp.dot(JFx, dW)

#%% Sampling in Tangent Space

class TMSampling(object):
    
    def __init__(self,
                 M:object,
                 x0:Tuple[Array, Array],
                 dim:int,
                 Exp_map:Callable[[Tuple[Array,Array], Array], Array]=None,
                 repeats:int=2**3,
                 x_samples:int=2**5,
                 t_samples:int=2**7,
                 N_sim:int=2**8,
                 max_T:float=1.0,
                 dt_steps:int=1000,
                 T_sample:bool = False,
                 t:float=0.1
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
            
            if not self.T_sample:
                inds = jnp.array(random.sample(range(self._dts.shape[0]), self.t_samples))
                ts = ts[inds]
                samples = xss[inds]
                
                yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1)),
                                  samples.reshape(-1,self.dim),
                                  jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                                  dW[inds].reshape(-1,self.dim),
                                  jnp.repeat(self._dts[inds],self.N_sim).reshape((-1,1)),
                                  ))
            
            else:
                inds = jnp.argmin(jnp.abs(ts-self.t))
                ts = ts[inds]
                samples = xss[inds]
                yield jnp.hstack((jnp.repeat(Fx0s,self.x_samples,axis=0),
                                  samples.reshape(-1,self.dim),
                                  jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                                  dW[inds].reshape(-1,self.dim),
                                  jnp.repeat(self._dts[inds],self.N_sim).reshape((-1,1)),
                                  ))
            
    def update_coords(self, Fx:Array)->Tuple[Array,Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (self.M.invF((Fx,chart)), Fx)
    
    def grad_TM(self,
                  s1_model:Callable[[Array, Array, Array], Array], 
                  x0:Array, 
                  x:Array, 
                  t:Array
                  )->Array:
        
        return self.M.proj(x, s1_model(x0, x, t))
    
    def grad_local(self,
                  s1_model:Callable[[Array, Array, Array], Array], 
                  x0:Array, 
                  x:Array, 
                  t:Array
                  )->Array:
        
        return jnp.dot(self.M.invJF((x[1],x[1])), s1_model(x0, x[1], t))
    
    def proj_hess(self,s1_model:Callable[[Array, Array, Array], Array], 
                   s2_model:Callable[[Array, Array, Array], Array],
                   x0:Array, 
                   x:Array, 
                   t:Array
                   )->Array:
        
        x = self.update_coords(x)

        Fx = self.M.F(x)
        
        JFx = self.M.JF(x)        
        
        val1 = self.M.proj(x0, s2_model(x0,Fx,t))
        val2 = s1_model(x0,Fx,t)-self.M.proj(x0, s1_model(x0,Fx,t))
        val3 = jacfwd(lambda x: self.M.proj(x, val2))(x0)
        
        return val1+val3#jnp.einsum('i,j->ij', M.proj(x0, s1_model(x0,Fx,t)), x)
    
    def dW_TM(self,
              x:Array,
              dW:Array
              )->Array:
        
        return dW#self.M.proj(x,dW)
    
    def dW_local(self,
                x:Array,
                dW:Array
                )->Array:
        
        return jnp.dot(self.M.invJF((x[1],x[1])), dW)
    
    def dW_embedded(self,
                x:Array,
                dW:Array
                )->Array:
        
        return dW

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
                  s1_model:Callable[[Array, Array, Array], Array], 
                  x0:Array, 
                  x:Array, 
                  t:Array
                  )->Array:
        
        return self.M.proj(x, s1_model(x0, x, t))
    
    def grad_local(self,
                  s1_model:Callable[[Array, Array, Array], Array], 
                  x0:Array, 
                  x:Tuple[Array,Array], 
                  t:Array
                  )->Array:
        
        return jnp.dot(self.M.invJF((x[1],x[1])), s1_model(x0, x[1], t))
    
    def proj_hess(self,s1_model:Callable[[Array, Array, Array], Array], 
                   s2_model:Callable[[Array, Array, Array], Array],
                   x0:Array, 
                   x:Array, 
                   t:Array
                   )->Array:
        
        x = self.update_coords(x)

        Fx = self.M.F(x)
        JFx = self.M.JF(x)        
        
        val1 = self.M.proj(x0, s2_model(x0,Fx,t))
        val2 = s1_model(x0,Fx,t)-self.M.proj(x0, s1_model(x0,Fx,t))
        val3 = jacfwd(lambda x: self.M.proj(x, val2))(x0)
        
        
        return val1+val3#jnp.einsum('i,j->ij', M.proj(x0, s1_model(x0,Fx,t)), x)
    
    def dW_TM(self,
              x:Array,
              dW:Array
              )->Array:
        
        return dW#self.M.proj(x,dW)
    
    def dW_local(self,
                x:Tuple[Array, Array],
                dW:Array
                )->Array:
        
        return jnp.dot(self.M.invJF(x), dW)
    
    def dW_embedded(self,
                x:Array,
                dW:Array
                )->Array:
        
        return dW

#%% VAE Sampling

class VAESampling(object):
    
    def __init__(self,
                 F:Callable[[Array],Array],
                 x0:Array,
                 dim:int,
                 method:str='Local',
                 repeats:int=2**3,
                 x_samples:int=2**5,
                 t_samples:int=2**7,
                 N_sim:int=2**8,
                 max_T:float=1.0,
                 dt_steps:int=1000,
                 )->None:
        
        self.F = F
        self.x_samples=x_samples
        self.dim = dim
        self.t_samples = t_samples
        self.N_sim = N_sim
        self.max_T = max_T
        self.dt_steps = dt_steps
        self.repeats = repeats
        if x0.ndim == 1:
            self.x0s = jnp.tile(x0, repeats)
        else:
            self.x0s = x0
        self.x0s_default = self.x0s
        dt = self.dts(T=self.max_T, n_steps=self.dt_steps)
        self.dt = dt
        self.t_grid = jnp.cumsum(dt)
        self.dt_tile = jnp.tile(dt, (self.N_sim,1))
        
        return
        
    def __str__(self)->str:
        
        return "Generating Samples for Brownian Motion on Manifolds in Local Coordinates for VAE"
    
    def dts(self, T:float=1.0,n_steps:int=n_steps)->Array:
        """time increments, deterministic"""
        return jnp.array([T/n_steps]*n_steps)

    def dWs(self,d:int,_dts:Array=None,num:int=1)->Array:
        """
        standard noise realisations
        time increments, stochastic
        """
        keys = jrandom.split(self.key,num=num+1)
        self.key = keys[0]
        subkeys = keys[1:]
        if _dts == None:
            _dts = self.dts()
        if num == 1:
            return jnp.sqrt(_dts)[:,None]*jrandom.normal(subkeys[0],(_dts.shape[0],d))
        else:
            return vmap(lambda subkey: jnp.sqrt(_dts)[:,None]*jrandom.normal(subkey,(_dts.shape[0],d)))(subkeys) 
        
    def Jf(self,z):
        
        return jacfwd(lambda z: self.F(z))(z)
        
    def G(self,z):
        
        Jf = self.Jf(z)
        
        return jnp.dot(Jf.T,Jf)
    
    def DG(self,z):
        
        return jacfwd(self.G)(z)
    
    def Ginv(self,z):
        
        return jnp.linalg.inv(self.G(z))
    
    def Chris(self,z):
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def taylor_sample(self):
        
        def sample(carry, step):
            
            t,z = carry
            dt, dW = step
            
            t += dt
            ginv = self.Ginv(z)
            
            stoch = jnp.dot(ginv, dW)
            
            z += stoch
            
            return ((t,z),)*2

        dW = dWs(self.N_sim*self.M.dim,self.dt).reshape(-1,self.dt_steps,self.M.dim)
        x0 = jnp.repeat(self.x0s, self.x_samples, axis=0)
        
        _, val =lax.scan(lambda carry, step: vmap(lambda t,z,dt,dW: sample((t,z),(dt,dW)))\
                         (carry[0],carry[1],step[0],step[1]),
                         init=(jnp.zeros(self.N_sim),x0), xs=(self.dt,dW)
                         )
        t,z = val

        return t, z, dW
    
    def local_sample(self)->Array:
        
        def sample(z, step):

            dt, t, dW = step
            
            t += dt
            ginv = self.Ginv(z)
            Chris = self.Chris(z)
            
            stoch = jnp.dot(ginv, dW)
            det = 0.5*jnp.einsum('jk,ijk->i', ginv, Chris)
            
            z += det+stoch
            
            t = t.astype(jnp.float32)
            z = z.astype(jnp.float32)
            
            return (z,)*2
        
        dW = self.dWs(self.N_sim*self.M.dim,self.dt).reshape(-1,self.dt_steps,self.M.dim)
        x0 = jnp.repeat(self.x0s, self.x_samples, axis=0)
        
        _, z =lax.scan(lambda carry, step: vmap(lambda t,z,dt,dW: sample((t,z),(dt,dW)))\
                         (carry[0],carry[1],step[0],step[1]),
                         init=(jnp.zeros(self.N_sim),x0), xs=(self.dt,self.t_grid, dW)
                         )

        dW = jnp.transpose(dW, axis=(1,0,2))

        return self.t_grid, z, dW
    
    def dts(self, T:float=1.0,n_steps:int=n_steps)->Array:
        """time increments, deterministic"""
        return jnp.array([T/n_steps]*n_steps)

    def dWs(self,d:int,_dts:Array=None,num:int=1)->Array:
        """
        standard noise realisations
        time increments, stochastic
        """
        keys = jrandom.split(self.key,num=num+1)
        self.key = keys[0]
        subkeys = keys[1:]
        if _dts == None:
            _dts = self.dts()
        if num == 1:
            return jnp.sqrt(_dts)[:,None]*jrandom.normal(subkeys[0],(_dts.shape[0],d))
        else:
            return vmap(lambda subkey: jnp.sqrt(_dts)[:,None]*jrandom.normal(subkey,(_dts.shape[0],d)))(subkeys) 
        
    def Jf(self,z):
        
        return jacfwd(lambda z: self.decoder(z.reshape(-1,2)).reshape(-1))(z)
        
    def G(self,z):
        
        Jf = self.Jf(z)
        
        return jnp.dot(Jf.T,Jf)
    
    def DG(self,z):
        
        return jacfwd(self.G)(z)
    
    def Ginv(self,z):
        
        return jnp.linalg.inv(self.G(z))
    
    def Chris(self,z):
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
        
    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:
            
            if self.method == 'Taylor':
                t,x,dW = self.taylor_sample()
            else:
                t,x,dW = self.local_sample()

            self.x0s = x[-1,::self.x_sampels]
            
            if jnp.isnan(jnp.sum(x)):
                self.x0s = self.x0s_default

            inds = jnp.array(random.sample(range(self.dt.shape[0]), self.t_samples))
            ts = ts[inds]
            samples = xss[inds]
            
            yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,self.x_samples,axis=0),(self.t_samples,1)),
                              samples.reshape(-1,self.M.dim),
                              jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                              dW[inds].reshape(-1,self.M.dim),
                              jnp.repeat(self._dts[inds],self.N_sim).reshape((-1,1)),
                              ))

    def update_coords(self, Fx:Array)->Tuple[Array,Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (Fx,chart)
    
    def grad_TM(self,
                  s1_model:Callable[[Array, Array, Array], Array], 
                  x0:Array, 
                  x:Array, 
                  t:Array
                  )->Array:
        
        return s1_model(x0, x, t)
    
    def grad_local(self,
                   s1_model:Callable[[Array, Array, Array], Array], 
                   x0:Array, 
                   x:Tuple[Array,Array], 
                   t:Array
                   )->Array:
        
        return s1_model(x0, x[0], t)
    
    def proj_hess(self,s1_model:Callable[[Array, Array, Array], Array], 
                   s2_model:Callable[[Array, Array, Array], Array],
                   x0:Array, 
                   x:Array, 
                   t:Array
                   )->Array:
        
        return s2_model(x0,x,t)
    
    def dW_TM(self,
                x:Array,
                dW:Array
                )->Array:
    
        return dW
    
    def dW_local(self,
                x:Array,
                dW:Array
                )->Array:
    
        return dW
    
    def dW_embedded(self,
                x:Array,
                dW:Array
                )->Array:
        
        return dW


