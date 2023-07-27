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

from src.setup import *
from src.params import *

from src.manifolds.manifold import *

from src.plotting import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

class Ellipsoid(EmbeddedManifold):
    """ 2d Ellipsoid """
    
    def __str__(self):
        
        return "Circle S1"
    
    def __init__(self, 
                 angle_shift:float = 2*jnp.pi,
                 use_spherical_coords = True):
        self.use_spherical_coords = use_spherical_coords
        self.angle_shift = angle_shift
        
        self.F_spherical = lambda x: jnp.arctan2(x[1][1]/x[1][0]) % self.angle_shift
        self.F_spherical_inv = lambda x: jnp.array([jnp.cos(x[0]), jnp.sin(x[1])])
        
        self.F_steographic = lambda x: x[1][0]/(1-x[1][1])
        self.F_steographic_inv = lambda x: jnp.array([2*x[0], x[0]**2-1])/(u[0]**2+1)
        
        do_chart_update_spherical = lambda x: x[0] > self.angle_shift-.1 
        do_chart_update_steographic = lambda x: x[0] > .1 # look for a new chart if true
        
        if use_spherical_coords:
            F = self.F_spherical
            invF = self.F_spherical_inv
            self.do_chart_update = do_chart_update_spherical
            self.centered_chart = self.centered_chart_spherical
            self.StdLog = self.StdLogSpherical
        else:
            F = self.F_steographic
            invF = self.F_steographic_inv
            self.do_chart_update = do_chart_update_steographic
            self.centered_chart = self.centered_chart_steographic
            self.Stdlog = self.StdLogSteographic
            
        F = self.F_steographic
        invF = self.F_steographic_inv

        EmbeddedManifold.__init__(self,F,1,2,invF=invF)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        self.acts = lambda g,x: jnp.tensordot(g,x,(2,0))

    def chart(self):
        """ return default coordinate chart """
        
        return jnp.array([0.0, 0.1])
    
    def centered_chart_spherical(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return jax.lax.stop_gradient(x[0] % (2*jnp.pi))
        else:
            return x % (2*jnp.pi) # already in embedding space

    def centered_chart_steographic(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return jax.lax.stop_gradient(self.F(x))
        else:
            return x # already in embedding space

    def StdLogSpherical(self, x,y):

        return (y[0]-x[0])%self.angle_shift
    
    def StdLogEmb(self, x,y):
        proj = lambda x,y: jnp.dot(x,y)*x
        Fx = self.F(x)
        v = y-proj(Fx,y)
        theta = jnp.arccos(jnp.dot(Fx,y))
        normv = jnp.linalg.norm(v,2)
        w = jax.lax.cond(normv >= 1e-5,
                         lambda _: theta/normv*v,
                         lambda _: jnp.zeros_like(v),
                         None)
    
    def StdLogSteographic(self, x,y):
        Fx = self.F(x)
        return jnp.dot(self.invJF((Fx,x[1])),self.StdLogEmb(x,y))
