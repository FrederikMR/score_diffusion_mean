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
from jaxgeometry.integration import integrate_sde, integrator_stratonovich

#%% Brownian coords

def brownian_projection(M:object)->None:
    """ Brownian motion as a submanifold in R^n"""

    def sde_brownian_projection(c:Tuple[Array, Array, Array],
                                y:Tuple[Array, Array]
                                )->Tuple[Array, Array, Array, float]:
        
        t,x,chart,s = c
        dt,dW = y
        
        X = vmap(lambda v: M.proj(x, v))(jnp.eye(M.emb_dim))
        det = jnp.zeros(M.emb_dim)
        sto = jnp.tensordot(X,dW,(1,0))
        
        return (det,sto,X,0.)
    
    def chart_update_brownian_projection(x:Array,
                                         chart:Array,
                                         *ys
                                         ):
        
        return (x, M.invF((x,x)), *ys)
    
    def brownian_projection(x,dts,dWs,stdCov=1.):
        
        val = integrate_sde(sde_brownian_projection,
                            integrator_stratonovich,
                            chart_update_brownian_projection,
                            x[0],x[1],dts,dWs,stdCov)[0:3]
        
        return (val[1],val[0],*val[2:])
    
    if hasattr(M, "proj"):
    
        M.sde_brownian_projection = sde_brownian_projection
        M.chart_update_brownian_projection = chart_update_brownian_projection
        
        M.brownian_projection = jit(brownian_projection)
    else:
        print("The manifold does not have a 'proj' attribute")
    
    return
