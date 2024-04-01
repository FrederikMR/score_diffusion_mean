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
from .nEllipsoid import nEllipsoid
from .riemannian import Manifold, metric, curvature, geodesic, Log, parallel_transport

#%% n-Sphere

class nSphere(nEllipsoid):
    """ n-Sphere """
    
    def __init__(self, N:int=2, use_spherical_coords:bool=False,chart_center:int=None):
        
        if chart_center is None:
            chart_center = N
        
        nEllipsoid.__init__(self,
                           N = N,
                           params=jnp.ones(N+1,dtype=jnp.float32),
                           chart_center=chart_center,
                           use_spherical_coords=use_spherical_coords)
        
        self.hk = jit(lambda x,y,t, N_terms=100: hk(self, x,y, t, N_terms))
        self.hk_embedded = lambda x,y,t, N_terms=100: hk_embedded(self, x,y, t, N_terms)
        self.log_hk = jit(lambda x,y,t, N_terms=100: log_hk(self, x, y, t, N_terms))
        self.gradx_log_hk = jit(lambda x,y,t, N_terms=100: gradx_log_hk(self, x, y, t, N_terms))
        self.grady_log_hk = jit(lambda x,y,t, N_terms=100: grady_log_hk(self, x, y, t, N_terms))
        #self.ggrady_log_hk = jit(lambda x,y,t: -jnp.eye(self.dim)/t)
        self.gradt_log_hk = jit(lambda x,y,t, N_terms=100: gradt_log_hk(self, x, y, t, N_terms))
    
    def __str__(self):
        return "%dd sphere (ellipsoid parameters %s, spherical_coords: %s)" % (self.dim,self.params,self.use_spherical_coords)

#%% Heat Kernels

def hk(M:object, x:Array,y:Array,t:float, N_terms=100) -> float:
    
    def sum_term(l:int, C_l:float) -> float:
    
        return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:Tuple[float,float,float], l:int) -> Tuple[Tuple[float, float, float], None]:

        val, Cl1, Cl2 = carry

        C_l = update_cl(l, Cl1, Cl2)

        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None

    x1 = x[1]
    y1 = y[1]
    xy_dot = jnp.dot(x1,y1)
    m1 = M.dim-1
    Am_inv = jnp.exp(jscipy.special.gammaln((M.dim+1)*0.5))/(2*jnp.pi**((M.dim+1)*0.5))
    
    alpha = m1*0.5
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(0, C_0) + sum_term(1, C_1)
    
    grid = jnp.arange(2,N_terms,1)
    
    val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
        
    return val[0]*Am_inv/m1

def hk_embedded(M:object, x:Array,y:Array,t:float, N_terms=100) -> float:
    
    def sum_term(l:int, C_l:float) -> float:
    
        return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:Tuple[float,float,float], l:int) -> Tuple[Tuple[float, float, float], None]:

        val, Cl1, Cl2 = carry

        C_l = update_cl(l, Cl1, Cl2)

        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None

    x1 = x
    y1 = y
    xy_dot = jnp.dot(x1,y1)
    m1 = M.dim-1
    Am_inv = jnp.exp(jscipy.special.gammaln((M.dim+1)*0.5))/(2*jnp.pi**((M.dim+1)*0.5))
    
    alpha = m1*0.5
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(0, C_0) + sum_term(1, C_1)
    
    grid = jnp.arange(2,N_terms,1)
    val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
        
    return val[0]*Am_inv/m1

def log_hk(M:object, x:Array,y:Array,t:float, N_terms=100) -> float:
    
    return jnp.log(hk(M,x,y,t, N_terms))

def gradx_log_hk(M:object, x:Array, y:Array, t:float, N_terms=100) -> float:
    
    def get_coords(Fx:Array) -> Tuple[Array, Array]:

        chart = M.centered_chart(Fx)
        return (M.invF((Fx,chart)),chart)

    def to_TM(Fx:Array, v:Array) -> Array:
        
        x = get_coords(Fx)
        JFx = M.JF(x)
        
        return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

    def to_TMx(Fx:Array,v:Array) -> Array:

        x = get_coords(Fx)

        return jnp.dot(M.invJF((Fx,x[1])),v)
    
    def sum_term(l:int, C_l1:float) -> float:
    
        return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l1
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:Tuple[float, float, float], l:int) -> Tuple[Tuple[float, float, float], None]:
        
        val, Cl1, Cl2 = carry
        
        C_l = update_cl(l-1, Cl1, Cl2)
        
        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None

    x1 = x[1]
    y1 = y[1]
    
    xy_dot = jnp.dot(x1, y1)
    alpha = (M.dim+1)*0.5
    m1 = M.dim-1
    Am_inv = jnp.exp(jscipy.special.gammaln((M.dim+1)*0.5))/(2*jnp.pi**((M.dim+1)*0.5))
    
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(1, C_0)+sum_term(2, C_1)
    
    grid = jnp.arange(3,N_terms,1)
    
    val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
    
    grad = val[0]*y1*Am_inv/hk(M,x,y,t,N_terms)
    
    return (to_TMx(x1, grad), to_TM(x1, grad))

def grady_log_hk(M:object, x:Array, y:Array, t:float, N_terms=100) -> float:
    
    def get_coords(Fx:Array) -> Tuple[Array, Array]:

        chart = M.centered_chart(Fx)
        return (M.invF((Fx,chart)),chart)

    def to_TM(Fx:Array, v:Array) -> Array:
        
        x = get_coords(Fx)
        JFx = M.JF(x)
        
        return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

    def to_TMx(Fx:Array,v:Array) -> Array:

        x = get_coords(Fx)

        return jnp.dot(M.invJF((Fx,x[1])),v)
    
    def sum_term(l:int, C_l1:float) -> float:
    
        return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l1
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:Tuple[float, float, float], l:int)->Tuple[Tuple[float, float, float], None]:
        
        val, Cl1, Cl2 = carry
        
        C_l = update_cl(l-1, Cl1, Cl2)
        
        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None

    x1 = x[1]
    y1 = y[1]
    
    xy_dot = jnp.dot(x1, y1)
    alpha = (M.dim+1)*0.5
    m1 = M.dim-1
    Am_inv = jnp.exp(jscipy.special.gammaln((M.dim+1)*0.5))/(2*jnp.pi**((M.dim+1)*0.5))
    
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(1, C_0)+sum_term(2, C_1)
    
    grid = jnp.arange(3,N_terms,1)
    
    val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
    
    grad = val[0]*x1*Am_inv/hk(M,x,y,t,N_terms)
    
    return (to_TMx(y1, grad), to_TM(y1, grad))

def gradt_log_hk(M:object, x:Array, y:Array, t:float, N_terms=100) -> float:
    
    def sum_term(l:int, C_l:float) -> float:
    
        return -0.5*l*(l+m1)*jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:Tuple[float, float, float], l:int) -> Tuple[Tuple[float, float, float], None]:

        val, Cl1, Cl2 = carry

        C_l = update_cl(l, Cl1, Cl2)

        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None
    
    x1 = x[1]
    y1 = y[1]
    
    xy_dot = jnp.dot(x1,y1)
    m1 = M.dim-1
    Am_inv = jnp.exp(jscipy.special.gammaln((M.dim+1)*0.5))/(2*jnp.pi**((M.dim+1)*0.5))
    
    alpha = m1*0.5
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(0, C_0) + sum_term(1, C_1)
    
    grid = jnp.arange(2,N_terms,1)
    
    val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
        
    return val[0]*Am_inv/(m1*hk(M,x,y,t,N_terms))

#%% Old code

"""
def get_coords(Fx:Array) -> tuple[Array, Array]:

    chart = M.centered_chart(Fx)
    return (M.invF((Fx,chart)),chart)

def to_TM(Fx:Array, v:Array) -> Array:
    
    x = get_coords(M, Fx)
    JFx = M.JF(x)
    
    return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

def to_TMchart(Fx:Array,v:Array) -> Array:
    
    x = get_coords(M, Fx)
    JFx = M.JF(x)
    return jnp.dot(JFx,v)

def to_TMx(Fx:Array,v:Array) -> Array:

    x = get_coords(M, Fx)

    return jnp.dot(M.invJF((Fx,x[1])),v)
"""