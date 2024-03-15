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
        self.x0s = tile(x0, repeats)
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


