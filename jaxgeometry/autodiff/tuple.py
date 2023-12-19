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

#%% Automatic Differentiation for tuple containing coordinate and chart

def gradx(f:Callable[[Tuple[Array, Array]], Array]):
    """ jax.grad but only for the x variable of a function taking coordinates and chart """
    def fxchart(x:Array,chart:Array,*args,**kwargs)->Array:
        return f((x,chart),*args,**kwargs)
    def gradf(x:Tuple[Array, Array],*args,**kwargs)->Array:
        return grad(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return gradf

def jacfwdx(f:Callable[[Tuple[Array, Array]], Array]):
    """jax.jacfwd but only for the x variable of a function taking coordinates and chart"""
    def fxchart(x:Array,chart:Array,*args,**kwargs)->Array:
        return f((x,chart),*args,**kwargs)
    def jacf(x:Tuple[Array, Array],*args,**kwargs)->Array:
        return jacfwd(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return jacf

def jacrevx(f:Callable[[Tuple[Array, Array]], Array]):
    """jax.jacrev but only for the x variable of a function taking coordinates and chart"""
    def fxchart(x:Array,chart:Array,*args,**kwargs):
        return f((x,chart),*args,**kwargs)
    def jacf(x:Tuple[Array, Array],*args,**kwargs)->Array:
        return jacrev(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return jacf

def hessianx(f:Callable[[Tuple[Array, Array]], Array])->Array:
    """hessian only for the x variable of a function taking coordinates and chart"""
    return jacfwdx(jacrevx(f))

def straight_through(f:Callable[[Tuple[Array, Array]], Array],
                     x:Tuple[Array, Array],
                     *ys)->Tuple[Array, Array]:
    """
    evaluation with pass through derivatives
    Create an exactly-zero expression with Sterbenz lemma that has
    an exactly-one gradient.
    """
    if type(x) == type(()):
        #zeros = tuple([xi - stop_gradient(xi) for xi in x])
        fx = lax.stop_gradient(f(x,*ys))
        return tuple([fxi - lax.stop_gradient(fxi) for fxi in fx])
    else:
        return x-lax.stop_gradient(x)
        #zero = x - stop_gradient(x)
        #return zeros + stop_gradient(f(x,*ys))