#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:43:33 2023

@author: frederik
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *
import jaxgeometry.manifolds.riemannian as riemannian

#%% Euclidean Geometry (R^n)

class Euclidean(riemannian.Manifold):
    """ Euclidean space """

    def __init__(self,N:int=3)->None:
        riemannian.Manifold.__init__(self)
        self.dim = N

        self.do_chart_update = lambda x: False
        self.update_coords = lambda coords,_: coords

        ##### Metric:
        self.g = lambda x: jnp.eye(self.dim)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        
        riemannian.metric(self)
        riemannian.curvature(self)
        riemannian.geodesic(self)
        riemannian.Log(self)
        riemannian.parallel_transport(self)
        
        #Metric
        self.Gamma_g = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim)))
        self.DGamma_g = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim, self.dim)))
        self.gsharp = jit(lambda x: jnp.eye(self.dim))
        self.Dg = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim)))
        self.mu_Q = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        self.det = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        self.detsharp = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        self.logAbsDet = jit(lambda x: jnp.zeros(1, dtype=jnp.float32))
        self.logAbsDetsharp = jit(lambda x: jnp.zeros(1, dtype=jnp.float32))
        self.dot = jit(lambda x,v,w: v.dot(w))
        self.dotsharp = jit(lambda x, p, pp: pp.dot(p))
        self.flat = jit(lambda x,v: v)
        self.sharp = jit(lambda x,p: p)
        self.orthFrame = jit(lambda x: jnp.eye(self.dim))
        self.div = jit(lambda x,X: jnp.trace(jacfwdx(X)(x)))
        self.divsharp = jit(lambda x,X: jnp.trace(jacfwdx(X)(x)))
        
        #Geodesic
        self.geodesic = jit(lambda x,v,dts: (jnp.cumsum(dts), jnp.stack((x[0]+jnp.cumsum(dts)[:,None]*v, 
                                                                         jnp.tile(v, (len(dts), 1)))).transpose(1,0,2), 
                                             jnp.tile(x[1], (len(dts), 1))))
        
        #Log
        self.Log = jit(lambda x,y: (y[0]-x[0], jnp.zeros(1)))
        self.dist = jit(lambda x,y: jnp.sqrt(jnp.sum((x[0]-y[0])**2)))
        
        #Parallel Transport - ADD CLOSED FORM EXPRESSIONS
        
        #Curvature - ADD CLOSED FORM EXPRESSIONS
        
        
        #ADD HEAT KERNEL
        self.hk = jit(lambda x,y,t: hk(self, x,y, t))
        self.log_hk = jit(lambda x,y,t: log_hk(self, x, y, t))
        self.gradx_log_hk = jit(lambda x,y,t: gradx_log_hk(self, x, y, t))
        self.grady_log_hk = jit(lambda x,y,t: grady_log_hk(self, x, y, t))
        self.gradt_log_hk = jit(lambda x,y,t: gradt_log_hk(self, x, y, t))
        self.mlx_hk = jit(lambda X_obs,t: mlx_hk(self, X_obs, t))
        self.mlt_hk = jit(lambda X_obs,t: mlt_hk(self, X_obs, t))
        self.mlxt_hk = jit(lambda X_obs, t: mlxt_hk(self, X_obs))
        
        return
    
    def update_vector(self, coords:ndarray, new_coords:ndarray, new_chart:ndarray, v:ndarray)->ndarray:
        
        return v

    def __str__(self)->str:
        
        return "Euclidean manifold of dimension %d" % (self.dim)

    def plot(self)->None:
        if self.dim == 2:
            plt.axis('equal')
    
    def plot_path(self, xs:tuple[ndarray, ndarray], 
                  u:ndarray=None, 
                  color:str='b', 
                  color_intensity:float=1., 
                  linewidth:float=1., 
                  prevx:tuple[ndarray, ndarray]=None, 
                  last:bool=True, 
                  s:int=20, 
                  arrowcolor:str='k'
                  )->None:
        
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            self.plotx(x, u=u if i == 0 else None,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       s=s,
                       prevx=prevx,
                       last=i==N-1)
            prevx = x
            
        return

    def plotx(self, x:tuple[ndarray, ndarray], 
              u:ndarray=None, 
              color:str='b', 
              color_intensity:float=1., 
              linewidth:float=1., 
              prevx:tuple[ndarray, ndarray]=None,
              last:bool=True, 
              s:int=20, 
              arrowcolor:str='k'
              )->None:
        assert(type(x) == type(()) or x.shape[0] == self.dim)
        if type(x) == type(()):
            x = x[0]
        if type(prevx) == type(()):
            prevx = prevx[0]

        ax = plt.gca()

        if last:
            if self.dim == 2:
                plt.scatter(x[0],x[1],color=color,s=s)
            elif self.dim == 3:
                ax.scatter(x[0],x[1],x[2],color=color,s=s)
        else:
            try:
                xx = np.stack((prevx,x))
                if self.dim == 2:
                    plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)
                elif self.dim == 3:
                    ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
            except:
                if self.dim == 2:
                    plt.scatter(x[0],x[1],color=color,s=s)
                elif self.dim == 3:
                    ax.scatter(x[0],x[1],x[2],color=color,s=s)

        try:
            plt.quiver(x[0], x[1], u[0], u[1], pivot='tail', linewidth=linewidth, scale=5, color=arrowcolor)
        except:
            pass
        
        return
        
#%% Heat Kernel

def hk(M:Euclidean, x:ndarray,y:ndarray,t:ndarray)->ndarray:
    
    const = 1/((2*jnp.pi*t)**(self.dim*0.5))
    
    return jnp.exp(-0.5*jnp.sum(x[0]-y[0])/t)*const

def log_hk(M:Euclidean, x:ndarray,y:ndarray,t:ndarray)->ndarray:
    
    return -0.5*jnp.sum(x[0]-y[0])/t-self.dim*0.5*jnp.log(2*jnp.pi*t)

def gradx_log_hk(M:Euclidean, x:ndarray, y:ndarray, t:ndarray)->float:
    
    return ((y[0]-x[0])/t, jnp.zeros(1))

def grady_log_hk(M:Euclidean, x:ndarray, y:ndarray, t:ndarray)->float:
    
    return ((x[0]-y[0])/t, jnp.zeros(1))

def gradt_log_hk(M:Euclidean, x:ndarray, y:ndarray, t:ndarray)->float:
    
    diff = x[0]-y[0]
    
    return 0.5*jnp.dot(diff, diff)/(t**2)-0.5*self.dim/t

def mlx_hk(M:Euclidean, X_obs:ndarray, t:ndarray=None)->float:

    return (jnp.mean(X_obs[0], axis=0), jnp.zeros(1))

def mlt_hk(M:Euclidean, X_obs:ndarray, mu:ndarray)->float:

    diff_mu = X_obs[0]-mu[0]
    
    return jnp.mean(jnp.linalg.norm(diff_mu, axis = 1)**2)/self.dim

def mlxt_hk(M:Euclidean, X_obs:ndarray)->float:
    
    mu = hk_mu(X_obs)
    
    return mu, opt_t(X_obs, mu)



































