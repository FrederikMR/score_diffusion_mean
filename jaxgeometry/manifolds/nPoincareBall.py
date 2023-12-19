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

#https://arxiv.org/pdf/2003.00335.pdf

#%% Modules

from jaxgeometry.setup import *
from .riemannian import Manifold
from jaxgeometry.operators.Gyration import *

#%% Euclidean Geometry (R^n)

class nPoincareBall(Manifold):
    """ n Poincare Ball """
    
    def __str__(self)->str:
        
        return f"{self.dim}-dimensioanl Poincare Ball with curvate {self.K} embedded into R^{self.dim}"

    def __init__(self,N:int=3, inner_product:str='Lorentzian', K:float=-1.)->None:
        Manifold.__init__(self)
        self.dim = N
        self.K = K

        self.do_chart_update = lambda x: False
        self.update_coords = lambda coords,_: coords

        ##### Metric:
        self.lambdaK = lambda x: 2/(1+self.K*jnp.dot(x[0],x[0]))
        self.g = lambda x: jnp.eye(self.dim)*self.lambdaK(x)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        
        self.dot = lambda x,v,w: jnp.tensordot(jnp.tensordot(M.g(x),w,(1,0)),v,(0,0))
        self.norm = lambda x,v: jnp.sqrt(self.dot(x,v,v))
        self.dist = lambda x,y: jnp.arccosh(1-2*jnp.dot(x[0]-y[0])/(1+K*jnp.dot(x[0],x[0])*jnp.dot(y[0],y[0]))) / \
            jnp.sqrt(jnp.abs(self.K))
        self.Exp = lambda x,v,T=1.0: MobiusAddition(x[0],
                                                    jnp.tanh(jnp.sqrt(jnp.abs(self.K))*self.lambdaK(x)*jnp.dot(v,v)*0.5) * \
                                                        v/(jnp.sqrt(self.abs(K))*jnp.dot(v,v)),
                                                    self.K)
        self.Log = lambda x,y: LogMap(self, x, y)
        self.ParallelTransport = lambda x,y,v: v-self.K*self.dot(y[0],v)*(x[0]+y[0])/(1+self.K*self.dot(x[0],y[0]))
        
        return
    
    def update_vector(self, coords:ndarray, new_coords:ndarray, new_chart:ndarray, v:ndarray)->ndarray:
        
        return v



