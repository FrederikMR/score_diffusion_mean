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
from .riemannian import Manifold

#%% Euclidean Geometry (R^n)

class Grassmanian(Manifold):
    """ Euclidean space """

    def __init__(self,N:int=3, K:int=3)->None:
        Manifold.__init__(self)
        
        self.N = N
        self.K = K
        self.dim = N*K
        
        return

    def __str__(self)->str:
        
        return "Grassmanian Manifold %d" % (self.dim)
    
    def Exp(self, x:Tuple[Array, Array], v:Array, T=1.0)->Tuple[Array, Array]:
        
        U,S,V = jnp.linalg.svd(T*v)
        
        return (jnp.dot(jnp.dot(jnp.dot(x[0], V), jnp.cos(S)), V.T) + \
            jnp.dot(jnp.dot(U, jnp.sin(S)), V.T), jnp.zeros(1))
            
    def Log(self, x:Tuple[Array, Array], y:Tuple[Array, Array]):
        
        x1 = x[0]
        y1 = y[0]
        
        A = jnp.linalg.solve(jnp.dot(y1.T,x1), y.T-jnp.dot(y.T, jnp.dot(x, x.T)))
        
        U,S,V = jnp.linalg.svd(A)
        
        return jnp.dot(V, jnp.dot(jnp.arctan(S), U.T))
    
    def dot(self, x:Tuple[Array, Array], v:Array, w:Array) -> Array:
        
        return jnp.trace(jnp.dot(v.T, w))
    
    def norm(self, x:Tuple[Array, Array], v:Array) -> Array:
        
        return jnp.sqrt(jnp.sum(v**2))
    
    def dist(self, x:Tuple[Array, Array], y:Tuple[Array, Array])->Array:
        
        A = jnp.dot(x[0].T, y[0])
        
        U,S,V = jnp.linalg.svd(A)
        
        b = jnp.arccos(S)
        
        b = b.at[S>=1].set(0)
        
        return jnp.abs(b)
    
    def Proj(self, x:Tuple[Array, Array], v:Array):
        
        return v-jnp.dot(x[0], jnp.dot(x[0].T, v))
    
    def ParallelTransport(self, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:
        
        return self.Proj(y,v)
        



















