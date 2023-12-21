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

#%% Integration

def dts(T:float=T,n_steps:int=n_steps)->Array:
    """time increments, deterministic"""
    return jnp.array([T/n_steps]*n_steps)

def dWs(d:int,_dts:Array=None,num:int=1)->Array:
    """
    standard noise realisations
    time increments, stochastic
    """
    global key
    keys = jrandom.split(key,num=num+1)
    key = keys[0]
    subkeys = keys[1:]
    if _dts == None:
        _dts = dts()
    if num == 1:
        return jnp.sqrt(_dts)[:,None]*jrandom.normal(subkeys[0],(_dts.shape[0],d))
    else:
        return vmap(lambda subkey: jnp.sqrt(_dts)[:,None]*jrandom.normal(subkey,(_dts.shape[0],d)))(subkeys) 
    
def StdNormal(d:int,num:int=1)->Array:
    """
    standard noise realisations
    time increments, stochastic
    """
    global key
    keys = jrandom.split(key,num=num+1)
    key = keys[0]
    subkeys = keys[1:]
    return jrandom.normal(subkeys[0],(d,num)).squeeze()

def integrator(ode_f:Callable[[Tuple[Array, Array, Array], Array], Array],
               chart_update:Callable[[Array, Array, Array], Tuple[Array, Array]]=None,
               method:str=default_method
               )->Callable:
    """
    Integrator (deterministic)
    """
    if chart_update == None: # no chart update
        chart_update = lambda *args: args[0:2]

    # euler:
    def euler(c:Tuple[Array, Array, Array],y:Tuple[Array, Array]):
        t,x,chart = c
        dt,*_ = y
        return ((t+dt,*chart_update(x+dt*ode_f(c,y[1:]),chart,y[1:])),)*2

    # Runge-kutta:
    def rk4(c:Tuple[Array, Array, Array],y:Tuple[Array, Array]):
        t,x,chart = c
        dt,*_ = y
        k1 = ode_f(c,y[1:])
        k2 = ode_f((t+dt/2,x + dt/2*k1,chart),y[1:])
        k3 = ode_f((t+dt/2,x + dt/2*k2,chart),y[1:])
        k4 = ode_f((t,x + dt*k3,chart),y[1:])
        return ((t+dt,*chart_update(x + dt/6*(k1 + 2*k2 + 2*k3 + k4),chart,y[1:])),)*2

    if method == 'euler':
        return euler
    elif method == 'rk4':
        return rk4
    else:
        assert(False)

def integrate(ode:Callable[[Tuple[Array, Array, Array], Array], Array],
              chart_update:Callable[[Array, Array, Array], Tuple[Array, Array]],
              x:Array,
              chart:Array,
              dts:Array,
              *ys
              )->Array:
    """return symbolic path given ode and integrator"""

    _,xs = lax.scan(integrator(ode,chart_update),
                    (0.,x,chart),
                    (dts,*ys))
    return xs if chart_update is not None else xs[0:2]

def integrate_sde(sde:Callable[[Tuple[Array, Array, Array, Array], Tuple[Array, Array]], Tuple[Array, Array, Array, Array]],
                  integrator:Callable,
                  chart_update:Callable[[Array, Array, Array], Tuple[Array, Array]],
                  x:Array,
                  chart:Array,
                  dts:Array,
                  dWs:Array,
                  *cy
                  )->Array:
    """
    sde functions should return (det,sto,Sigma) where
    det is determinisitc part, sto is stochastic part,
    and Sigma stochastic generator (i.e. often sto=dot(Sigma,dW)
    """
    _,xs = lax.scan(integrator(sde,chart_update),
                    (0.,x,chart,*cy),
                    (dts,dWs,))
    return xs

def integrator_stratonovich(sde_f:Callable[[Tuple[Array, Array, Array, Array], Tuple[Array, Array]], Tuple[Array, Array, Array, Array]],
                            chart_update:Callable[[Array, Array, Array], Tuple[Array, Array]]=None
                            )->Callable:
    """Stratonovich integration for SDE"""
    if chart_update == None: # no chart update
        chart_update = lambda xp,chart,*cy: (xp,chart,*cy)

    def euler_heun(c,y):
        t,x,chart,*cy = c
        dt,dW = y

        (detx, stox, X, *dcy) = sde_f(c,y)
        tx = x + stox
        cy_new = tuple([y+dt*dy for (y,dy) in zip(cy,dcy)])
        return ((t+dt,*chart_update(x + dt*detx + 0.5*(stox + sde_f((t+dt,tx,chart,*cy),y)[1]), chart, *cy_new),),)*2

    return euler_heun

def integrator_ito(sde_f:Callable[[Tuple[Array, Array, Array, Array], Tuple[Array, Array]], Tuple[Array, Array, Array, Array]],
                   chart_update:Callable[[Array, Array, Array], Tuple[Array, Array]]=None,
                   step_fun:Callable[[Array, Array], Array]=lambda x,v: x[0]+v
                   )->Callable:
    
    """Ito integration for SDE"""
    
    if chart_update == None: # no chart update
        chart_update = lambda xp,chart,*cy: (xp,chart,*cy)

    def euler(c,y):
        t,x,chart,*cy = c
        dt,dW = y

        (detx, stox, X, *dcy) = sde_f(c,y)
        cy_new = tuple([y+dt*dy for (y,dy) in zip(cy,dcy)])
        return ((t+dt,*chart_update(step_fun((x,chart), dt*detx + stox), chart, *cy_new)),)*2

    return euler

def integrator_ito_tm(sde_f:Callable[[Tuple[Array, Array, Array, Array], Tuple[Array, Array]], Tuple[Array, Array, Array, Array]],
                      chart_update:Callable[[Array, Array, Array], Tuple[Array, Array]]=None,
                      step_fun:Callable[[Array, Array], Array]=lambda x,v: x[0]+v
                      )->Callable:
    
    """Ito integration for SDE"""
    
    if chart_update == None: # no chart update
        chart_update = lambda xp,chart,*cy: (xp,chart,*cy)

    def euler(c,y):
        t,x,chart,*cy = c
        dt,dW = y

        (detx, stox, X,*dcy) = sde_f(c,y)
        cy_new = tuple([y+dt*dy for (y,dy) in zip(cy,dcy)])
        return ((t+dt,*chart_update(x, step_fun((x,chart), dt*detx + stox), *cy_new)),)*2

    return euler

def integrate_sde_tm(sde:Callable[[Tuple[Array, Array, Array, Array], Tuple[Array, Array]], Tuple[Array, Array, Array, Array]],
                  integrator:Callable,
                  chart_update:Callable[[Array, Array, Array], Tuple[Array, Array]],
                  x:Array,
                  chart:Array,
                  dts:Array,
                  dWs:Array,
                  *cy
                  )->Array:
    """
    sde functions should return (det,sto,Sigma) where
    det is determinisitc part, sto is stochastic part,
    and Sigma stochastic generator (i.e. often sto=dot(Sigma,dW)
    """

    _,xs = lax.scan(integrator(sde,chart_update),
                    (0.,x,chart,*cy),
                    (dts,dWs))
    return xs