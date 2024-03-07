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

class Stiefel(EmbeddedManifold):
    """ Stiefiel Manifold """
    
    def __str__(self)->str:
        
        return f"Stiefiel({self.N}, {self.K})"

    def __init__(self,
                 N:int=3,
                 K:int=2,
                 )->None:
        EmbeddedManifold.__init__(self,F=self.F,dim=N*K-K*(K+1)//2,emb_dim=N*K,invF=self.invF)
        self.N = N
        self.K = K

        self.do_chart_update = lambda x: False
        
        self.update_coords = lambda coords,_: coords

        ##### Metric:

        self.dot = self.StdDot
        self.norm = self.StdNorm
        self.Exp = self.StdExp
        self.ExpEmbedded = self.StdExpEmbedded
        self.proj = self.StdProj
        
        return
    
    def chart(self):
        """ return default coordinate chart """
        return jnp.block([jnp.eye(self.K), jnp.zeros((self.K,self.N-self.K))]).T.reshape(-1)

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
    
    def StdExp(self, x:Tuple[Array, Array], v:Array, T:float=1.0)->Tuple[Array, Array]:
        
        Fx = x[1].reshape(self.N, self.K)
        v = v.reshape(self.N, self.K)
        
        val1 = jnp.dot(Fx.T, v)
        val2 = -jnp.dot(v.T,v)
        
        mat1 = jnp.block([Fx, T*v])
        mat2 = jnp.block([[val1, -val2],
                          [jnp.eye(self.K), val1]
                          ])
        mat3 = jnp.block([jscipy.linalg.expm(-val1), jnp.zeros((self.K, self.K))]).T
        
        exp_map = jnp.dot(mat1, jnp.dot(jscipy.linalg.expm(mat2), mat3))
        
        return (exp_map.reshape(-1), exp_map.reshape(-1))
            
    def StdExpEmbedded(self, x:Array, v:Array, T:float=1.0)->Tuple[Array, Array]:
        
        Fx = x.reshape(self.N, self.K)
        v = v.reshape(self.N, self.K)
        
        val1 = jnp.dot(Fx.T, v)
        val2 = -jnp.dot(v.T,v)
        
        
        mat1 = jnp.block([Fx, T*v])
        mat2 = jnp.block([[val1, -val2],
                          [jnp.eye(self.K), val1]
                          ])
        mat3 = jnp.block([jscipy.linalg.expm(-val1), jnp.zeros((self.K, self.K))]).T
        
        exp_map = jnp.dot(mat1, jnp.dot(jscipy.linalg.expm(mat2), mat3))
        
        return exp_map.reshape(-1)
    
    def StdDot(self, x:Tuple[Array, Array], v:Array, w:Array) -> Array:
        
        return jnp.trace(jnp.dot(v.reshape(self.N,self.K).T, w.reshape(self.N,self.K)))
    
    def StdNorm(self, x:Tuple[Array, Array], v:Array) -> Array:
        
        return jnp.sqrt(jnp.sum(v**2))
    
    def StdProj(self, x:Array, v:Array):
        
        Fx = x.reshape(self.N,self.K)
        v = v.reshape(self.N,self.K)
        
        val1 = jnp.dot(Fx.T, v)
        B = 0.5*jnp.dot(val1.T,val1)

        return (v-jnp.dot(Fx,B)).reshape(-1)
    
    def ParallelTransport(self, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:
        
        return self.StdProj(y[1], v)