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

#https://arxiv.org/pdf/2003.00335.pdf

#%% Modules

from jaxgeometry.setup import *

#%% Code

def MobiusAddition(x:Array,
                   y:Array,
                   K:float=1.
                   )->Array:
    
    xy_dot = jnp.dot(x,y)
    normx2 = jnp.dot(x,x)
    normy2 = jnp.dot(y,y)
    
    term1 = (1-K*(2*xy_dot-normy2))*x
    term2 = (1+K*normx2)*y
    term3 = 1-K*(2*xy_dot+K*normx2*normy2)
    
    return (term1+term2)/term3

def MobiusSubstraction(x:Array,
                       y:Array,
                       K:float=1.
                       )->Array:
    
    return MobiusAddition(x,-y,K)

def GyrationOperator(x:Array,
                     y:Array,
                     v:Array,
                     K:float=1.
                     )->Array:
    
    zeros = jnp.zeros_like(x)
    
    term1 = MobiusAddition(x,y)
    term2 = MobiusAddition(y,v)
    term3 = MobiusAddition(x,term2)
    term4 = MobiusAddition(term1, term3)
    term5 = MobiusSubstraction(zeros, term4)
    
    return term5