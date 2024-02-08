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

#https://indico.ictp.it/event/a08167/session/124/contribution/85/material/0/0.pdf
#https://www.cis.upenn.edu/~cis6100/geomean.pdf
#https://poisson.phc.dm.unipi.it/~maxreen/bruno/pdf/D.%20Bini%20and%20B.%20Iannazzo%20-%20A%20note%20on%20computing%20Matrix%20Geometric%20Means.pdf
#https://manoptjl.org/v0.1/manifolds/hyperbolic/
#https://proceedings.neurips.cc/paper_files/paper/2020/file/1aa3d9c6ce672447e1e5d0f1b5207e85-Paper.pdf

#%% Modules

from jaxgeometry.setup import *
from .riemannian import EmbeddedManifold, metric, curvature, geodesic, Log, parallel_transport

#%% Symmetric Positive Definite Space in Local Coordinates

class Sym(EmbeddedManifold):
    """ manifold of symmetric positive definite matrices """

    def __init__(self,N=3):
        self.N = N
        
        idx, col = jnp.triu_indices(self.N,k=0)
        self.tri = jnp.zeros((self.N,self.N))
        self.idx = idx
        self.col = col
        
        dim = N*(N+1)//2
        emb_dim = N*N
        EmbeddedManifold.__init__(self,
                                  F=self.F,
                                  dim=dim,
                                  emb_dim=emb_dim, 
                                  invF=self.invF)

        self.act = lambda g,q: jnp.tensordot(g,jnp.tensordot(q.reshape((N,N)),g,(1,1)),(1,0)).flatten()
        self.acts = vmap(self.act,(0,None))
        
        self.do_chart_update = lambda x: False
        
        metric(self)
        curvature(self)
        geodesic(self)
        Log(self)
        parallel_transport(self)
        
        self.proj = self.StdProj
        self.Expt = self.StdExpt
        self.Exp = lambda x,v: self.Expt(x,v,t=1.0)
        self.ExpEmbedded = self.ExpEmbedded
        self.Log = self.StdLog
        self.dist = self.StdDist
        self.ParallelTransport = self.StdParallelTransport

    def __str__(self):
        return "Sym(%d), dim %d" % (self.N,self.dim)
    
    def F(self, x:Tuple[Array, Array])->Array:
        
        l = self.tri
        l = l.at[self.idx,self.col].set(x[0].reshape(-1))
        
        return (l+l.T-jnp.diag(jnp.diag(l))).reshape(-1)
    
    def invF(self, x:Tuple[Array, Array])->Array:
        
        return (x[0].reshape(self.N,self.N)[self.idx,self.col]).reshape(-1)
    
    def centered_chart(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return lax.stop_gradient(self.F(x))
        else:
            return x#self.F((x,jnp.zeros(self.N*self.N))) # already in embedding space
        
    def chart(self):
        """ return default coordinate chart """
        return jnp.eye(self.N).reshape(-1)
    
    def StdExpt(self, x:Tuple[Array, Array], v:Array, t:float=1.0)->Array:
        
        Fx = self.F(x)#.reshape(self.N,self.N)
        Fv = jnp.dot(self.JF(x),v)
        
        w = Fx+Fv
        
        return (self.invF((w, w)), w.reshape(-1))
    
    def StdProj(self, x:Array, v:Array) -> Array:
        
        v = v.reshape(self.N, self.N)
        
        return 0.5*(v+v.T).reshape(-1)
    
    def ExpEmbedded(self, Fx:Array, v:Array, t:float=1.0)->Array:
        
        v = self.StdProj(Fx, v)
        
        return Fx+v
    
    def StdLog(self, x:Tuple[Array, Array], y:Array)->Array:
        
        w = y-x[1]
            
        return jnp.dot(self.invJF((x[1],x[1])),w)
    
    def StdParallelTransport(self, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:
        
        return v
    
    def StdDist(self, x:Tuple[Array, Array], y:Tuple[Array,Array])->Array:
        
        return jnp.linalg.norm(self.F(x)-self.F(y), 'fro')