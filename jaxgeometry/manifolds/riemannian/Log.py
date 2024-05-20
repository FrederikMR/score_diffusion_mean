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
from jaxgeometry.integration import dts

#%% Logaritmic Map

def Log(M:object,
               f=None,
               method='BFGS'
               )->None:
    """ numerical Riemannian Logarithm map """

    def loss(x:Tuple[Array, Array],
             v:Array,
             y:Tuple[Array, Array]
             )->float:
        
        (x1,chart1) = f(x,v)
        y_chart1 = M.update_coords(y,chart1)
        
        return 1./M.dim*jnp.sum(jnp.square(x1 - y_chart1[0]))
    
    def shoot(x:Tuple[Array, Array],
              y:Tuple[Array, Array],
              v0:Array=None
              )->Tuple[Array, Array]:

        if v0 is None:
            v0 = jnp.zeros(M.dim)
        #res = jopt.minimize(lambda w: loss(x,w,y), v0, method=method, options={'maxiter': 100})
        res = minimize(lambda w: (loss(x,w,y),dloss(x,w,y)), v0, method=method, jac=True, options={'disp': False, 'maxiter': 100})
        #res = jscipy.optimize.minimize(lambda w: (loss(x,w,y),dloss(x,w,y)), v0, method=method, 
        #                               options={'maxiter': 100})

        return res.x.squeeze()#(res.x,res.fun)
    
    def dist(x:Tuple[Array, Array],
             y:Tuple[Array, Array]
             )->float:
        
        v = M.Log(x,y)
        
        curve = M.geodesic(x,v[0],dts(T,n_steps))
        
        dt = jnp.diff(curve[0])
        val = vmap(lambda v: M.norm(x,v))(curve[1][:,1])
        
        return jnp.sum(0.5*dt*(val[1:]+val[:-1])) #trapezoidal rule

    if f is None:
        print("using M.Exp for Logarithm")
        f = M.Exp

    dloss = grad(loss,1)

    M.Log = shoot
    M.dist = dist
    
    return
