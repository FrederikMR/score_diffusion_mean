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
from .riemannian import EmbeddedManifold, metric, curvature, geodesic, Log, parallel_transport

#%% Latent Manifold

class HypParaboloid(EmbeddedManifold):
    """ Hyperbolic Paraboloid """

    def __init__(self):
        
        F = lambda x: jnp.array([x[0][0], x[0][1], x[0][0]**2-x[0][1]**2])
        invF = lambda x: jnp.array([x[1][0], x[1][0]])
        
        EmbeddedManifold.__init__(self,F,2,3,invF=invF)

        # metric matrix
        self.g = lambda x: jnp.dot(self.JF(x).T,self.JF(x))
        self.update_coords = lambda x,y: x
        
        metric(self)
        curvature(self)
        geodesic(self)
        Log(self)
        parallel_transport(self)
        
    def __str__(self):
        
        return "Hyperbolic Paraboloid"