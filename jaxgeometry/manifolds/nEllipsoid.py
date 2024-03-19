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
from jaxgeometry.operators.vectors import *
from jaxgeometry.integration import dWs, StdNormal

#%% Ellipsoid

class nEllipsoid(EmbeddedManifold):
    """ N-d Ellipsoid """
    
    def __init__(self,
                 N:int=2,
                 params:Array=None,
                 chart_center:int=None,
                 use_spherical_coords:bool=False):
        
        if params is None:
            params = jnp.ones(N+1, dtype=jnp.float32)
        if chart_center is None:
            chart_center = N
        
        self.params = jnp.array(params) # ellipsoid parameters (e.g. [1.,1.,1.] for sphere)
        self.use_spherical_coords = use_spherical_coords
        self.chart_center = chart_center
        
        def F_steographic(x):
            
            s2 = jnp.sum(x[0]**2)
            val = jnp.concatenate(((1-s2).reshape(1), 2*x[0]))/(1+s2)
                
            return self.params*jnp.dot(self.get_B(x[1]/self.params), val)
    
        def invF_steographic(x):
            
            Rinvx = jnp.linalg.solve(self.get_B(x[1]/self.params),x[0]/self.params)
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
            F = self.F_spherical
            invF = self.F_spherical_inv
        else:
            F = self.F_steographic
            invF = self.F_steographic_inv

        EmbeddedManifold.__init__(self,F,N,N+1,invF=invF)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        self.acts = lambda g,x: jnp.tensordot(g,x,(2,0))
        
        metric(self)
        curvature(self)
        geodesic(self)
        Log(self)
        parallel_transport(self)
        
        self.Log = self.StdLog
        self.Expt = self.StdExpt
        self.Exp = jit(lambda x,v: self.StdExpt(x,v,t=1.0))
        self.dist = self.StdDist
        self.dot = self.StdDot
        self.norm = self.StdNorm
        self.ParallelTransport = self.StdParallelTransport
        self.proj = self.StdProj
        
        return
    
    def __str__(self):
        return "%dd ellipsoid, parameters %s, spherical coords %s" % (self.dim,self.params,self.use_spherical_coords)

    def chart(self):
        """ return default coordinate chart """
        return jnp.eye(self.dim+1)[:,self.chart_center]/self.params

    def centered_chart(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return lax.stop_gradient(self.F(x))#/self.params
        else:
            return x#/self.params # already in embedding space

    def get_B(self,v):
        """ R^N basis with first basis vector v """
        if self.dim == 2:
            b1 = v
            k = jnp.argmin(jnp.abs(v))
            ek = jnp.eye(3)[:,k]
            b2 = ek-v[k]*v
            b3 = cross(b1,b2)
            return jnp.stack((b1,b2,b3),axis=1)
        else:
            b1 = v.reshape(-1,1)
            u, _, _ = jnp.linalg.svd(b1)
            bn = u[:,1:]
        
            return jnp.concatenate((b1, bn), axis=1)

    # Logarithm with standard Riemannian metric on S^n
    def StdLog(self, x:Tuple[Array, Array],y:Array):
        y = y/self.params # from ellipsoid to S^n
        proj = lambda x,y: jnp.dot(x,y)*x
        Fx = self.F(x)/self.params
        v = y-proj(Fx,y)
        theta = jnp.arccos(jnp.dot(Fx,y))
        normv = jnp.linalg.norm(v,2)
        w = lax.cond(normv >= 1e-5,
                     lambda _: theta/normv*v,
                     lambda _: jnp.zeros_like(v),
                     None)
        
        return jnp.dot(self.invJF((Fx,x[1])),self.params*w)
    
    # Logarithm with standard Riemannian metric on S^n
    def StdExpt(self, x:Tuple[Array, Array],v:Array,t:float=1.0):

        Fx = self.F(x)/self.params # from ellipsoid to S^n
        Fv = jnp.dot(self.JF(x),v)/self.params
        
        normv = jnp.linalg.norm(Fv*t,2)
        y = jnp.cos(normv)*Fx+Fv*jnp.sin(normv)/jnp.linalg.norm(Fv, 2)
        
        y *= self.params
        
        return (self.invF((y, y)), y)
    
    def ExpEmbedded(self, Fx:Array,Fv:Array,t:float=1.0):

        Fx /= self.params # from ellipsoid to S^n
        Fx = Fx/jnp.linalg.norm(Fx)
        Fv /= self.params
        
        normv = jnp.linalg.norm(Fv*t,2)
        y = jnp.cos(t*normv)*Fx+Fv*jnp.sin(t*normv)/jnp.linalg.norm(Fv, 2)
        y = y/jnp.linalg.norm(y)
        y *= self.params
        
        return y
    
    def ExpEmbeddedLocal(self, Fx:Array,Fv:Array,t:float=1.0):

        Fx /= self.params # from ellipsoid to S^n
        Fx = Fx/jnp.linalg.norm(Fx)
        Fv /= self.params
        
        normv = jnp.linalg.norm(Fv*t,2)
        y = jnp.cos(t*normv)*Fx+Fv*jnp.sin(t*normv)/jnp.linalg.norm(Fv, 2)
        y = y/jnp.linalg.norm(y)
        y *= self.params
        
        return (self.invF((y,y)), y)

    def StdDist(self, x:Tuple[Array, Array], y:Tuple[Array, Array])->Array:
        
        return jnp.arccos(jnp.dot(self.F(x),self.F(y)))
    
    def StdDot(self, x:Tuple[Array, Array], v:Array, w:Array)->Array:
        
        Fv = jnp.dot(self.JF(x),v)/self.params
        Fw = jnp.dot(self.JF(x),w)/self.params
        
        return jnp.dot(Fv,Fw)
    
    def StdNorm(self, x:Tuple[Array, Array], v:Array)->Array:
        
        return jnp.sqrt(self.dot(v,v))
    
    def StdParallelTransport(self, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:

        JFx = self.JF(x)
        Fx = self.F(x)/self.params
        
        logxy = jnp.dot(JFx, self.Log(x,self.F(y)))/self.params
        logyx = jnp.dot(self.JF(y), self.Log(y,self.F(x)))/self.params
        Fv = jnp.dot(JFx,v)/self.params
        
        w = Fv-jnp.dot(logxy, Fv)*(logxy+logyx)/self.dist(x,y)
        
        return jnp.dot(self.invJF((Fx,x[1])),self.params*w)
    
    def StdProj(self, Fx:Array, v:Array):
        Fx = Fx/self.params
        v /= self.params
        
        proj_mat = jnp.eye(self.emb_dim)-jnp.einsum('i,j->ij', Fx, Fx)
        
        return jnp.dot(proj_mat, v)*self.params
    
    def ProjdW(self, x:Tuple[Array, Array], dW:Array)->Array:
        
        return self.StdProj(x[1],dW)
        
    
    