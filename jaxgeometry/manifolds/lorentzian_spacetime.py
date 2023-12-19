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
from .riemannian import Manifold, metric, curvature, geodesic, Log, parallel_transport

#%% Euclidean Geometry (R^n)

class LorentzSpacetime(Manifold):
    """ Lorentzian Spacetime Metric """

    def __init__(self,
                 g:Callable[[Tuple[Array, Array, Array]], Array], #Metric tensor Riemannian manifold
                 N:int #dimension of Riemannian manifold, N
                 )->None:
        Manifold.__init__(self)
        self.dim = N+1
        

        self.do_chart_update = lambda x: False
        self.update_coords = lambda coords,_: coords

        ##### Metric:
        def SpacetimeMetric(x:Tuple[Array, Array]):
            
            G = -g(x)
            
            return jnp.block([[jnp.ones(1), jnp.zeros((1,N))],
                              [jnp.zeros((N,1)), G]])

        self.g = lambda x: SpacetimeMetric(x)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        
        metric(self)
        curvature(self)
        geodesic(self)
        Log(self)
        parallel_transport(self)
        
        #Metric
        #self.Gamma_g = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim)))
        #self.DGamma_g = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim, self.dim)))
        #self.gsharp = jit(lambda x: jnp.eye(self.dim))
        #self.Dg = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim)))
        #self.mu_Q = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        #self.det = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        #self.detsharp = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        #self.logAbsDet = jit(lambda x: jnp.zeros(1, dtype=jnp.float32))
        #self.logAbsDetsharp = jit(lambda x: jnp.zeros(1, dtype=jnp.float32))
        #self.dot = jit(lambda x,v,w: v.dot(w))
        #self.dotsharp = jit(lambda x, p, pp: pp.dot(p))
        #self.flat = jit(lambda x,v: v)
        #self.sharp = jit(lambda x,p: p)
        #self.orthFrame = jit(lambda x: jnp.eye(self.dim))
        #self.div = lambda x,X: jnp.trace(jacfwdx(X)(x))
        #self.divsharp = lambda x,X: jnp.trace(jacfwdx(X)(x))
        
        #Geodesic
        #self.geodesic = jit(lambda x,v,dts: (jnp.cumsum(dts), jnp.stack((x[0]+jnp.cumsum(dts)[:,None]*v, 
        #                                                                 jnp.tile(v, (len(dts), 1)))).transpose(1,0,2), 
        #                                     jnp.tile(x[1], (len(dts), 1))))
        
        #Log
        #self.Log = jit(lambda x,y: y[0]-x[0])
        #self.dist = jit(lambda x,y: jnp.sqrt(jnp.sum((x[0]-y[0])**2)))
        
        #Parallel Transport - ADD CLOSED FORM EXPRESSIONS
        
        #Curvature - ADD CLOSED FORM EXPRESSIONS
        
        return
    
    def update_vector(self, coords:Array, new_coords:Array, new_chart:Array, v:Array)->Array:
        
        return v

    def __str__(self)->str:
        
        return "Lorentzian spacetime manifold of %d" % (self.dim)


















