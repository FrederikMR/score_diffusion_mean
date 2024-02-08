#%% Euclidean Geometry (R^n)

## This file is part of Jax Geometry
#
# Copyright (C) 2021, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/jaxgeometry
#
# Jax Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Jax Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Jax Geometry. If not, see <http://www.gnu.org/licenses/>.
#

#%% Sources

#%% Modules

from jaxgeometry.setup import *
from .riemannian import EmbeddedManifold

#%% Euclidean Geometry (R^n)

class Grassmanian(EmbeddedManifold):
    """ Hyperbolic Space """
    
    def __str__(self)->str:
        
        return f"{self.dim}-dimensioanl Hyperbolic Space with curvate {self.K} embedded into R^{self.dim+1}"

    def __init__(self,N:int=3, K:int=3)->None:
        EmbeddedManifold.__init__(self,F=self.F,dim=N*K,emb_dim=N*K,invF=self.invF)
        self.N = N
        self.K = K

        self.do_chart_update = lambda x: False
        
        self.update_coords = lambda coords,_: coords

        ##### Metric:

        self.dot = self.StdDot
        self.norm = self.StdNorm
        self.dist = self.StdDist
        self.Exp = self.StdExp
        self.ExpEmbedded = self.StdExpEmbedded
        self.Log = self.StdLog
        self.proj = self.StdProj
        
        return
    
    def chart(self):
        """ return default coordinate chart """
        return jnp.eye(max(self.N,self.K))[:min(self.N,self.K)].reshape(-1)

    def centered_chart(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return lax.stop_gradient(self.F(x))
        else:
            return x # already in embedding space
    
    def F(self, x:Tuple[Array, Array])->Array:
        
        return x[0]
    
    def invF(self, x:Tuple[Array, Array])->Array:
        
        return x[1]
    
    def __str__(self)->str:
        
        return "Grassmanian Manifold %d" % (self.dim)
    
    def StdExp(self, x:Tuple[Array, Array], v:Array, T:float=1.0)->Tuple[Array, Array]:
        
        v = v.reshape(self.N,self.K)
        
        U,S,V = jnp.linalg.svd(T*v, full_matrices=False)
        S = jnp.diag(S)
        
        A = jnp.dot(jnp.dot(jnp.dot(x[1].reshape(self.N,self.K), V), jnp.cos(S)), V.T) + \
            jnp.dot(jnp.dot(U, jnp.sin(S)), V.T)
            
        exp_map, _ = jnp.linalg.qr(A)
        
        return (exp_map.reshape(-1), exp_map.reshape(-1))
            
    def StdExpEmbedded(self, x:Array, v:Array, T:float=1.0)->Tuple[Array, Array]:
        
        v = v.reshape(self.N,self.K)
        
        U,S,V = jnp.linalg.svd(T*v, full_matrices=False)
        S = jnp.diag(S)
                
        A = jnp.dot(jnp.dot(jnp.dot(x.reshape(self.N,self.K), V), jnp.cos(S)), V.T) + \
            jnp.dot(jnp.dot(U, jnp.sin(S)), V.T)
            
        exp_map, _ = jnp.linalg.qr(A)
        
        return exp_map.reshape(-1)
            
    def StdLog(self, x:Tuple[Array, Array], y:Array):
        
        x1 = x[1].reshape(self.N,self.K)
        y1 = y.reshape(self.N,self.K)
        
        A = jnp.linalg.solve(jnp.dot(y1.T,x1), y1.T-jnp.dot(y1.T, jnp.dot(x1, x1.T)))
        
        U,S,V = jnp.linalg.svd(A)
        
        return jnp.dot(V, jnp.dot(jnp.arctan(S), U.T)).reshape(-1)
    
    def StdDot(self, x:Tuple[Array, Array], v:Array, w:Array) -> Array:
        
        return jnp.trace(jnp.dot(v.reshape(self.N,self.K).T, w.reshape(self.N,self.K)))
    
    def StdNorm(self, x:Tuple[Array, Array], v:Array) -> Array:
        
        return jnp.sqrt(jnp.sum(v**2))
    
    def StdDist(self, x:Tuple[Array, Array], y:Tuple[Array, Array])->Array:
        
        A = jnp.dot(x[1].reshape(self.N,self.K).T, y[1].reshape(self.N,self.K))
        
        U,S,V = jnp.linalg.svd(A, full_matrices=False)
        
        b = jnp.arccos(S)
        
        return jnp.sqrt(jnp.sum(b[S>=1]**2))
    
    def StdProj(self, x:Array, v:Array):
        
        x1 = x.reshape(self.N,self.K)
        v = v.reshape(self.N,self.K)
        
        Y = v-jnp.dot(x1, jnp.dot(x1.T, v))
        
        return (Y/jnp.sqrt(jnp.sum(Y**2))).reshape(-1)
    
    def ParallelTransport(self, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:
        
        x1 = x[1].reshape(self.N,self.K)
        v = v.reshape(self.N,self.K)
        
        return (v-jnp.dot(x1, jnp.dot(x1.T, v))).reshape(-1)

