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
from jaxgeometry.integration import integrate_sde, integrator_ito

#%% Geodesic Random Walk

def GRW(M:object,
        b_fun:Callable[[float,Array, Array], Array] = None,
        sigma_fun:Callable[[float,Array, Array], Array] = None,
        f_fun:Callable[[Tuple[Array, Array], Array], Array] = None
        )->None:
    
    def sde_grw(c:Tuple[Array, Array, Array],
                y:Tuple[Array, Array]
                )->Tuple[Array, Array, Array, float]:
        
        t,x,chart,s = c
        dt,dW = y
        
        dW = M.proj(x,dW)
        
        det = b_fun(t, x, chart)
        X = sigma_fun(t, x, chart)
        sto = jnp.tensordot(X,dW,(1,0))
        
        return (det,sto,X,0.)
    
    def chart_update_grw(x:Tuple[Array, Array],
                         chart:Array,
                         *ys
                         ):
        
        return (x, M.invF((x,x)), *ys)
    
    if b_fun is None:
        b_fun = lambda t,x,v: jnp.zeros(M.emb_dim)
    if sigma_fun is None:
        sigma_fun = lambda t,x,v: jnp.eye(M.emb_dim)
    if f_fun is None:
        f_fun = lambda x,v: M.Exp(x,v)[0]
    
    if hasattr(M, "proj"):
    
        M.sde_grw = sde_grw
        M.chart_update_grw = chart_update_grw
        
        M.random_walk = jit(lambda x,dts,dWs, stdCov=1.: integrate_sde(sde_grw,
                                                                       lambda a,b: integrator_ito(a,b,lambda x,v: f_fun(x,v)),
                                                                       chart_update_grw,
                                                                       x[0],x[1],dts,dWs,stdCov)[0:3])
    else:
        print("The manifold does not have a 'proj' attribute")
    
    return