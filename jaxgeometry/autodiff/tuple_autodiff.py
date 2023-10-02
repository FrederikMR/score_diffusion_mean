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

def gradx(f:Callable[[Tuple[ndarray, ndarray], ...], ndarray])->Callable[[Tuple[ndarray, ndarray], ...], ndarray]:
    """ jax.grad but only for the x variable of a function taking coordinates and chart """
    def fxchart(x:ndarray,chart:ndarray,*args,**kwargs)->ndarray:
        return f((x,chart),*args,**kwargs)
    def gradf(x:Tuple[ndarray, ndarray],*args,**kwargs)->ndarray:
        return grad(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return gradf

def jacfwdx(f:Callable[[Tuple[ndarray, ndarray], ...], ndarray])->Callable[[Tuple[ndarray, ndarray], ...], ndarray]:
    """jax.jacfwd but only for the x variable of a function taking coordinates and chart"""
    def fxchart(x:ndarray,chart:ndarray,*args,**kwargs)->ndarray:
        return f((x,chart),*args,**kwargs)
    def jacf(x:Tuple[ndarray, ndarray],*args,**kwargs)->ndarray:
        return jacfwd(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return jacf

def jacrevx(f:Callable[[Tuple[ndarray, ndarray], ...], ndarray]):
    """jax.jacrev but only for the x variable of a function taking coordinates and chart"""
    def fxchart(x:ndarray,chart:ndarray,*args,**kwargs):
        return f((x,chart),*args,**kwargs)
    def jacf(x:Tuple[ndarray, ndarray],*args,**kwargs)->ndarray:
        return jacrev(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return jacf

def hessianx(f:Callable[[Tuple[ndarray, ndarray], ...], ndarray])->ndarray:
    """hessian only for the x variable of a function taking coordinates and chart"""
    return jacfwdx(jacrevx(f))

def straight_through(f:Callable[[Tuple[ndarray, ndarray], ...], ndarray],
                     x:Tuple[ndarray, ndarray],
                     *ys)->Tuple[ndarray, ndarray]:
    """
    evaluation with pass through derivatives
    Create an exactly-zero expression with Sterbenz lemma that has
    an exactly-one gradient.
    """
    if type(x) == type(()):
        #zeros = tuple([xi - stop_gradient(xi) for xi in x])
        fx = stop_gradient(f(x,*ys))
        return tuple([fxi - stop_gradient(fxi) for fxi in fx])
    else:
        return x-stop_gradient(x)
        #zero = x - stop_gradient(x)
        #return zeros + stop_gradient(f(x,*ys))