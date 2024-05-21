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
from jaxgeometry.autodiff import *

#%% Circle

class S1(EmbeddedManifold):
    """ 2d Ellipsoid """
    
    def __str__(self):
        
        return "Circle S1"
    
    def __init__(self, 
                 angle_shift:float = 2*jnp.pi):
        
        self.dim = 1
        
        self.angle_shift = angle_shift
        
        self.F = lambda x: jnp.array([jnp.cos(x[0]),jnp.sin(x[0])])
        self.invF = lambda x: jnp.arctan(x[1])
        self.chart = lambda : jnp.array([1.0,0.0])
        self.do_chart_update = lambda x: True
        self.update_coords = lambda coords, _: coords#(coords[0] % self.angle_shift, coords[1] % self.angle_shift)
        
        EmbeddedManifold.__init__(self,self.F,1,1,invF=self.invF)
        
        ##### Metric:
        self.g = lambda x: jnp.ones(1).reshape(1,1)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        self.acts = lambda g,x: jnp.tensordot(g,x,(2,0))
        
        metric(self)
        curvature(self)
        geodesic(self)
        Log(self)
        parallel_transport(self)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        self.acts = lambda g,x: jnp.tensordot(g,x,(2,0))
            
        self.Exp = lambda x,v: ((x[0]+v) % self.angle_shift,)*2
        self.ExpEmbedded = lambda x,v: (x[1]+v) % self.angle_shift
        self.Log = lambda x,y: (y-x[0]) % self.angle_shift
        self.dist = lambda x,y: jnp.abs(y[0]-x[0]) % self.angle_shift
        self.proj = lambda x,v: v % self.angle_shift
        

        #Heat kernels
        self.hk = jit(lambda x,y,t: hk(self, x,y, t))
        self.log_hk = jit(lambda x,y,t: log_hk(self, x, y, t))
        self.gradx_log_hk = jit(lambda x,y,t: gradx_log_hk(self, x, y, t))
        self.grady_log_hk = jit(lambda x,y,t: grady_log_hk(self, x, y, t))
        #self.ggrady_log_hk = jit(lambda x,y,t: -jnp.eye(self.dim)/t)
        self.gradt_log_hk = lambda x,y,t: gradt_log_hk(self, x, y, t)
    
    def StdLog(self, x:Tuple[Array, Array], y:Tuple[Array, Array])->Array:
        
        return (y[0]-x[0]) % self.angle_shift
    
    def StdExp(self, x:Tuple[Array, Array], v:Array)->Array:
        
        z = (y[0]+x[0]) % self.angle_shift

        return (z, self.F((z,x[1])))
    
    def StdDot(self, x:Tuple[Array, Array], v:Array, w:Array)->Array:
        
        return v*w
    
    def StdNorm(self, x:Tuple[Array, Array], v:Array)->Array:
        
        return jnp.abs(v)
    
    def dist(self, x:Tuple[Array, Array], y:Tuple[Array, Array])->Array:
        
        return jnp.abs((x[0]-y[0])%self.angle_shift)
    
    def ParallelTransport(self, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:
        
        return v
    
    def Proj(self, x:Tuple[Array, Array], v:Array)->Array:
        
        return v

#%% Heat Kernel

def hk(M:object, x:Array,y:Array,t:float,N_terms=20)->float:
    
    def step(carry:float, k:int)->Tuple[float, None]:
        
        carry += jnp.exp(-0.5*(2*jnp.pi*k+x1-y1)**2/t)
        
        return carry, None
    
    x1 = x[0]#jnp.arctan2(x[1][1],x[1][0]) % (2*jnp.pi)
    y1 = y[0]#jnp.arctan2(y[1][1],y[1][0]) % (2*jnp.pi)
    
    const = 1/jnp.sqrt(2*jnp.pi*t)
   
    val, _ = lax.scan(step, init=jnp.zeros(1), xs=jnp.arange(-N_terms+1,N_terms,1)) 
   
    return val*const

def log_hk(M:object, x:Array,y:Array,t:float)->float:
    
    return jnp.log(hk(x,y,t))

def gradx_log_hk(M:object, x:Array,y:Array,t:float, N_terms=20)->Tuple[Array, Array]:
    
    def step(carry:float, k:int)->Tuple[float, None]:
        
        term1 = 2*jnp.pi*k+x1-y1
        
        carry -= jnp.exp(-0.5*(term1**2)*tinv)*term1*tinv
        
        return carry, None

    const = 1/jnp.sqrt(2*jnp.pi*t)
        
    x1 = x[0]#jnp.arctan2(x[1][1]/x[1][0]) % (2*jnp.pi)
    y1 = y[0]#jnp.arctan2(y[1][1]/y[1][0]) % (2*jnp.pi)
    tinv = 1/t
   
    val, _ = lax.scan(step, init=jnp.zeros(1), xs=jnp.arange(0,N_terms,1)) 
    grad = val*const/hk(M,x,y,t)
   
    return grad#grad_x, grad_chart

def grady_log_hk(M:object, x:Array, y:Array, t:float, N_terms=20) -> Tuple[Array, Array]:
    
    def step(carry:float, k:int)->Tuple[float,None]:
        
        term1 = 2*jnp.pi*k+x1-y1
        
        carry += jnp.exp(-0.5*(term1**2)*tinv)*term1*tinv
        
        return carry, None

    const = 1/jnp.sqrt(2*jnp.pi*t)
    
    x1 = x[0]#jnp.arctan2(x[1][1],x[1][0]) % jnp.pi
    y1 = y[0]#jnp.arctan2(y[1][1],y[1][0]) % jnp.pi
    tinv = 1/t
    
    val, _ = lax.scan(step, init=jnp.zeros(1), xs=jnp.arange(-N_terms+1,N_terms,1)) 
    grad = val*const/hk(M,x,y,t)
   
    return grad

def gradt_log_hk(M:object, x:Array, y:Array, t:float, N_terms=20)->float:
    
    def step1(carry:float, k:int)->Tuple[float,None]:
        
        term1 = 0.5*(2*jnp.pi*k+x1-y1)**2
        
        carry += jnp.exp(-term1/t)*term1/(t**2)
        
        return carry, None
    
    def step2(carry:float, k:int)->Tuple[float,None]:
        
        carry += jnp.exp(-(0.5*(2*jnp.pi*k+x1-y1)**2)/t)
        
        return carry, None
    
    x1 = x[0]#jnp.arctan2(x[1][1],x[1][0]) % (2*jnp.pi)
    y1 = y[0]#jnp.arctan2(y[1][1],y[1][0]) % (2*jnp.pi)
        
    const1 = 1/jnp.sqrt(2*jnp.pi*t)
    const2 = -1/(2*jnp.sqrt(jnp.pi)*(t)**(3/2))
   
    val1, _ = lax.scan(step1, init=jnp.zeros(1), xs=jnp.arange(-N_terms+1,N_terms,1)) 
    val1 *= const1
    
    val2, _ = lax.scan(step2, init=jnp.zeros(1), xs=jnp.arange(-N_terms+1,N_terms,1)) 
    val2 *= const2
   
    return (val1+val2)/hk(M,x,y,t)