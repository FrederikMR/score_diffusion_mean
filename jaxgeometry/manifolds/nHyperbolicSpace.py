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

class nHyperbolicSpace(EmbeddedManifold):
    """ Hyperbolic Space """
    
    def __str__(self)->str:
        
        return f"{self.dim}-dimensioanl Hyperbolic Space with curvate {self.K} embedded into R^{self.dim+1}"

    def __init__(self,N:int=3, inner_product:str='Minkowski', K:float=-1.)->None:
        EmbeddedManifold.__init__(self,F=self.F,dim=N,emb_dim=N,invF=self.invF)
        self.dim = N
        self.K = K

        self.do_chart_update = lambda x: False
        self.update_coords = lambda coords,_: coords

        ##### Metric:
            
        metric = jnp.eye(self.dim)
        #if inner_product == 'Lorentizian':
        #    metric = metric.at[0,0].set(-1)
        #else:
        #    metric = metric.at[-1,-1].set(-1)
            
        #self.g = lambda x: metric
        
        #metric(self)
        #curvature(self)
        #geodesic(self)
        #Log(self)
        #parallel_transport(self)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        
        self.dot = lambda x,v,w: -w[-1]*v[-1]+jnp.dot(w[:-1],v[:-1])#jnp.tensordot(jnp.tensordot(self.g(x),w,(1,0)),v,(0,0))
        self.norm = lambda x,v: jnp.sqrt(jnp.max(jnp.array([self.dot(x,v,v),0.])))
        self.dist = lambda x,y: 1/jnp.sqrt(jnp.abs(self.K))*jnp.arccosh(self.K*self.dot(x[1],y[1]))
        self.Exp = self.StdExp
        self.ExpEmbedded = self.StdExpEmbedded
        self.Log = lambda x,y: (y[1]-K*jnp.dot(x[1],y[1])*x[1])*jnp.arccosh(self.K*jnp.dot(x[1],y[1]))/ \
            jnp.sinh(jnp.arccosh(self.K*jnp.dot(x[1],y[1])))
        self.ParallelTransport = lambda x,y,v: v-self.K*self.dot(y[1],v)*(x[1]+y[1])/(1+self.K*self.dot(x[1],y[1]))
        self.proj = self.StdProj
        
        return
    
    def chart(self):
        """ return default coordinate chart """
        return jnp.concatenate((jnp.zeros(self.dim-1), -1.*jnp.ones(1)))

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
    
    def StdProj(self, x:Tuple[Array, Array], v:Array)->Array:
        
        return v+self.dot(x,x,v)*v
            
    def StdExp(self, x:Tuple[Array, Array], v:Array, T:float=1.0)->Array:
        
         val = T*self.norm(x,v)*jnp.sqrt(jnp.abs(self.K))
         
         exp_map = jnp.cosh(val)+v*jnp.sinh(val)/(val)
         
         exp_map = jnp.where(jnp.isnan(exp_map).any(),
                             x[1],
                             exp_map)
         
         return (exp_map, exp_map)
         
    def StdExpEmbedded(self, x:Array, v:Array, T:float=1.0)->Array:
        
         val = T*self.norm(x,v)*jnp.sqrt(jnp.abs(self.K))
         
         exp_map = jnp.cosh(val)+v*jnp.sinh(val)/(val)
         
         exp_map = jnp.where(jnp.isnan(exp_map).any(),
                             x,
                             exp_map)
         
         return exp_map












