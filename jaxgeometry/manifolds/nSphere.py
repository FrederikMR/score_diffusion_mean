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
        
        self.hk = jit(lambda x,y,t: hk(self, x,y, t))
        self.log_hk = jit(lambda x,y,t: log_hk(self, x, y, t))
        self.gradx_log_hk = jit(lambda x,y,t: gradx_log_hk(self, x, y, t))
        self.grady_log_hk = jit(lambda x,y,t: grady_log_hk(self, x, y, t))
        #self.ggrady_log_hk = jit(lambda x,y,t: -jnp.eye(self.dim)/t)
        self.gradt_log_hk = jit(lambda x,y,t: gradt_log_hk(self, x, y, t))
    
    def __str__(self):
        return "%dd sphere (ellipsoid parameters %s, spherical_coords: %s)" % (self.dim,self.params,self.use_spherical_coords)

#%% Heat Kernels

def hk(M:object, x:jnp.ndarray,y:jnp.ndarray,t:float) -> float:
    
    def sum_term(l:int, C_l:float) -> float:
    
        return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:tuple[float,float,float], l:int) -> tuple[tuple[float, float, float], None]:

        val, Cl1, Cl2 = carry

        C_l = update_cl(l, Cl1, Cl2)

        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None

    x1 = x[1]
    y1 = y[1]
    xy_dot = jnp.dot(x1,y1)
    
    alpha = m1*0.5
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(0, C_0) + sum_term(1, C_1)
    
    grid = jnp.arange(2,N_terms,1)
    
    val, _ = scan(step, (val, C_1, C_0), xs=grid)
    
    m1 = M.dim-1
    Am_inv = jscipy((M.dim+1)*0.5)/(2*jnp.pi**((M.dim+1)*0.5))
        
    return val[0]*Am_inv/m1

def log_hk(M:object, x:jnp.ndarray,y:jnp.ndarray,t:float) -> float:
    
    return jnp.log(hk(x,y,t))

def gradx_log_hk(M:object, x:jnp.ndarray, y:jnp.ndarray, t:float) -> float:
    
    def get_coords(Fx:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

        chart = M.centered_chart(Fx)
        return (M.invF((Fx,chart)),chart)

    def to_TM(Fx:jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        x = get_coords(M, Fx)
        JFx = M.JF(x)
        
        return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

    def to_TMx(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:

        x = get_coords(M, Fx)

        return jnp.dot(M.invJF((Fx,x[1])),v)
    
    def sum_term(l:int, C_l1:float) -> float:
    
        return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l1
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:tuple[float, float, float], l:int) -> tuple[tuple[float, float, float], None]:
        
        val, Cl1, Cl2 = carry
        
        C_l = update_cl(l-1, Cl1, Cl2)
        
        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None

    x1 = x[1]
    y1 = y[1]
    
    xy_dot = jnp.dot(x1, y1)
    alpha = (M.dim+1)*0.5
    
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(1, C_0)+sum_term(2, C_1)
    
    grid = jnp.arange(3,N_terms,1)
    
    val, _ = scan(step, (val, C_1, C_0), xs=grid)
    
    grad = val[0]*y1*Am_inv/hk(x,y,t)
    
    m1 = M.dim-1
    Am_inv = jscipy((M.dim+1)*0.5)/(2*jnp.pi**((M.dim+1)*0.5))
    
    return (to_TMx(M, x1, grad), to_TM(M, x1, grad))

def grady_log_hk(M:object, x:jnp.ndarray, y:jnp.ndarray, t:float) -> float:
    
    def get_coords(Fx:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

        chart = M.centered_chart(Fx)
        return (M.invF((Fx,chart)),chart)

    def to_TM(Fx:jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        x = get_coords(M, Fx)
        JFx = M.JF(x)
        
        return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

    def to_TMx(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:

        x = get_coords(M, Fx)

        return jnp.dot(M.invJF((Fx,x[1])),v)
    
    def sum_term(l:int, C_l1:float) -> float:
    
        return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l1
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:tuple[float, float, float], l:int)->tuple[tuple[float, float, float], None]:
        
        val, Cl1, Cl2 = carry
        
        C_l = update_cl(l-1, Cl1, Cl2)
        
        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None

    x1 = x[1]
    y1 = y[1]
    
    xy_dot = jnp.dot(x1, y1)
    alpha = (M.dim+1)*0.5
    
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(1, C_0)+sum_term(2, C_1)
    
    grid = jnp.arange(3,N_terms,1)
    
    val, _ = scan(step, (val, C_1, C_0), xs=grid)
    
    grad = val[0]*x1*Am_inv/hk(x,y,t)
    
    m1 = M.dim-1
    Am_inv = jscipy((M.dim+1)*0.5)/(2*jnp.pi**((M.dim+1)*0.5))
    
    return (to_TMx(M, y1, grad), to_TM(M, y1, grad))

def gradt_log_hk(M:object, x:jnp.ndarray, y:jnp.ndarray, t:float) -> float:
    
    def sum_term(l:int, C_l:float) -> float:
    
        return -0.5*l*(l+m1)*jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l
    
    def update_cl(l:int, Cl1:float, Cl2:float) -> float:
        
        return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
    
    def step(carry:tuple[float, float, float], l:int) -> tuple[tuple[float, float, float], None]:

        val, Cl1, Cl2 = carry

        C_l = update_cl(l, Cl1, Cl2)

        val += sum_term(l, C_l)

        return (val, C_l, Cl1), None
    
    x1 = x[1]
    y1 = y[1]
    
    xy_dot = jnp.dot(x1,y1)
    
    alpha = m1*0.5
    C_0 = 1.0
    C_1 = 2*alpha*xy_dot
    
    val = sum_term(0, C_0) + sum_term(1, C_1)
    
    grid = jnp.arange(2,N_terms,1)
    
    m1 = M.dim-1
    Am_inv = jscipy((M.dim+1)*0.5)/(2*jnp.pi**((M.dim+1)*0.5))
    
    val, _ = scan(step, (val, C_1, C_0), xs=grid)
        
    return val[0]*Am_inv/(m1*hk(x,y,t))

#%% Old code

"""
def get_coords(Fx:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

    chart = M.centered_chart(Fx)
    return (M.invF((Fx,chart)),chart)

def to_TM(Fx:jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
    
    x = get_coords(M, Fx)
    JFx = M.JF(x)
    
    return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

def to_TMchart(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:
    
    x = get_coords(M, Fx)
    JFx = M.JF(x)
    return jnp.dot(JFx,v)

def to_TMx(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:

    x = get_coords(M, Fx)

    return jnp.dot(M.invJF((Fx,x[1])),v)
"""