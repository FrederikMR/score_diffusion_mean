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

class nEllipsoid(EmbeddedManifold):
    """ 2d Ellipsoid """
    
    def __str__(self):
        return "%dd ellipsoid, parameters %s, spherical coords %s" % (self.dim,self.params,self.use_spherical_coords)
    
    def __init__(self,N=2,params=np.array([1.,1.,1.]),chart_center=2,use_spherical_coords=False):
        
        self.params = jnp.array(params) # ellipsoid parameters (e.g. [1.,1.,1.] for sphere)
        self.use_spherical_coords = use_spherical_coords
        self.chart_center = chart_center
        
        def F_steographic(x):
            
            s2 = jnp.sum(x[0]**2)
            val = jnp.concatenate(((1-s2).reshape(1), 2*x[0]))/(1+s2)
                
            return self.params*jnp.dot(self.get_B(x[1]), val)
                    
        def invF_steographic(x):
            
            Rinvx = jnp.linalg.solve(self.get_B(x[1]),x[0]/self.params)
            x0 = Rinvx[0]
            
            val = vmap(lambda xi: xi/(1+x0))(Rinvx[1:])
            
            return val
            
        self.F_steographic = F_steographic
        self.F_steographic_inv = invF_steographic
        # spherical coordinates, no charts
        def F_spherical(phi):
            cosx = jnp.concatenate((jnp.cos(phi[0]), jnp.ones(1)))
            sinx = jnp.concatenate((jnp.ones(1), jnp.cumprod(jnp.sin(phi[0]))))
            
            val = vmap(lambda x,y: x*y)(cosx, sinx)
                
            return val*self.params
        
        def F_spherical_inv(x):
            
            
            sumx = jnp.flip(jnp.cumsum(jnp.flip(x[1])**2))
            val = vmap(lambda x,y: jnp.arccos(x/jnp.sqrt(sumx)))(x[1][:-2], sumx[:-2])
            val = jnp.concatenate((val,
                                   0.5*jnp.pi-jnp.arctan((x[1][-2]+jnp.sqrt(jnp.sum(x[1][-2:]**2)))/x[1][-1])))
            
            return val
        
        self.F_spherical = F_spherical
        self.F_spherical_inv = F_spherical_inv
        self.JF_spherical = lambda x: jnp.jacobian(self.F_spherical(x),x)
        self.g_spherical = lambda x: jnp.dot(self.JF_spherical(x).T,self.JF_spherical(x))
        self.mu_Q_spherical = lambda x: 1./jnp.nlinalg.Det()(self.g_spherical(x))

        ## optionally use spherical coordinates in chart computations
        #if use_spherical_coords:
        #    F = lambda x: jnp.dot(x[1],self.F_spherical(x[0]))
        self.do_chart_update = lambda x: jnp.linalg.norm(x[0]) > .1 # look for a new chart if true
        if use_spherical_coords:
            F = self.F_steographic
            invF = self.F_steographic_inv
        else:
            F = self.F_spherical
            invF = self.F_spherical_inv

        EmbeddedManifold.__init__(self,F,N,N+1,invF=invF)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        self.acts = lambda g,x: jnp.tensordot(g,x,(2,0))

    def chart(self):
        """ return default coordinate chart """
        return jnp.eye(self.dim+1)[:,self.chart_center]

    def centered_chart(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return jax.lax.stop_gradient(self.F(x))/self.params
        else:
            return x/self.params # already in embedding space

    def get_B(self,v):
        """ R^3 basis with first basis vector v """
        b1 = v.reshape(-1,1)
        u, _, _ = jnp.linalg.svd(b1)
        bn = u[:,1:]
        
        return jnp.concatenate((b1, bn), axis=1)

    # Logarithm with standard Riemannian metric on S^2
    def StdLogEmb(self, x,y):
        y = y/self.params # from ellipsoid to S^2
        proj = lambda x,y: jnp.dot(x,y)*x
        Fx = self.F(x)/self.params
        v = y-proj(Fx,y)
        theta = jnp.arccos(jnp.dot(Fx,y))
        normv = jnp.linalg.norm(v,2)
        w = jax.lax.cond(normv >= 1e-5,
                         lambda _: theta/normv*v,
                         lambda _: jnp.zeros_like(v),
                         None)
        return self.params*w
    def StdLog(self, x,y):
        Fx = self.F(x)/self.params
        return jnp.dot(self.invJF((Fx,x[1])),self.StdLogEmb(x,y))