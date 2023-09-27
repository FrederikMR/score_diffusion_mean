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
from jaxgeometry.manifolds import Ellipsoid

#%% n-Sphere

class nSphere(Ellipsoid):
    """ n-Sphere """
    
    def __init__(self, N:int=2, use_spherical_coords:bool=False,chart_center:int=None):
        
        if chart_center is None:
            chart_center = N
        
        Ellipsoid.__init__(self,
                           N = N,
                           params=jnp.ones(N+1,dtype=jnp.float32),
                           chart_center=chart_center,
                           use_spherical_coords=use_spherical_coords)
    
    def __str__(self):
        return "%dd sphere (ellipsoid parameters %s, spherical_coords: %s)" % (self.dim,self.params,self.use_spherical_coords)

