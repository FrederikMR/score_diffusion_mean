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
import jaxgeometry.manifolds.riemannian as riemannian

#%% Euclidean Geometry (R^n)

class LearnedManifold(riemannian.Manifold):
    """ Euclidean space """

    def __init__(self, g, N)->None:
        riemannian.Manifold.__init__(self)
        self.dim = N

        self.do_chart_update = lambda x: False
        self.update_coords = lambda coords,_: coords#(coords[0] % (2*jnp.pi), coords[1])

        ##### Metric:
        self.g = g

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        
        riemannian.metric(self)
        riemannian.curvature(self)
        riemannian.geodesic(self)
        riemannian.Log(self)
        riemannian.parallel_transport(self)
        
        return
    
    def update_vector(self, coords:Array, new_coords:Array, new_chart:Array, v:Array)->Array:
        
        return v

    def __str__(self)->str:
        
        return "Learned manifold of dimension %d" % (self.dim)


















