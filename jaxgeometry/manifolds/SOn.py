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

class SO(EmbeddedManifold):
    """ Hyperbolic Space """
    
    def __str__(self)->str:
        
        return f"{self.dim}-dimensioanl Special Orthogonal Group"

    def __init__(self,
                 N:int=3
                 )->None:
        EmbeddedManifold.__init__(self,F=self.F,dim=N*(N-1)//2,emb_dim=N**2,invF=self.invF)
        self.N = N

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
        return jnp.eye(self.N).reshape(-1)

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
        
        return "Special Orthogonal Group %d" % (self.dim)
    
    def StdExp(self, x:Tuple[Array, Array], v:Array, T:float=1.0)->Tuple[Array, Array]:
        
        v = v.reshape(self.N,self.N)
        Fx = x[1].reshape(self.N,self.N)
        
        exp_map = jnp.dot(Fx, jscipy.linalg.expm(T*v))
        
        return (exp_map.reshape(-1), exp_map.reshape(-1))
            
    def StdExpEmbedded(self, x:Array, v:Array, T:float=1.0)->Tuple[Array, Array]:
        
        v = v.reshape(self.N,self.N)
        
        Fx = x.reshape(self.N,self.N)
        
        exp_map = jnp.dot(Fx, jscipy.linalg.expm(T*v))
        
        return exp_map.reshape(-1)
            
    def StdLog(self, x:Tuple[Array, Array], y:Array):
        
        Fx = x[1].reshape(self.N,self.N)
        Fy = y.reshape(self.N,self.N)
        prod = jnp.einsum('i,j->ij', Fx, Fy)
        log_val = logm(prod)
        
        return 0.5*(log_val-log_val.T).reshape(-1)
    
    def StdDot(self, x:Tuple[Array, Array], v:Array, w:Array) -> Array:
        
        return jnp.trace(jnp.dot(v.reshape(self.N,self.N).T, w.reshape(self.N,self.N)))
    
    def StdNorm(self, x:Tuple[Array, Array], v:Array) -> Array:
        
        return jnp.sqrt(jnp.sum(v**2))
    
    def StdDist(self, x:Tuple[Array, Array], y:Tuple[Array, Array])->Array:
        
        log_val = self.StdLog(x, y[1])
        
        return self.StdNorm(x, log_val)
    
    def StdProj(self, x:Array, v:Array):
        
        Fx = x.reshape(self.N,self.N)
        v = v.reshape(self.N,self.N)
        
        val = jnp.dot(Fx.T, v)
        
        return 0.5*(val-val.T).reshape(-1)
    
    def ParallelTransport(self, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:
        
        return v.reshape(-1)

