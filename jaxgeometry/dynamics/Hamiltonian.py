#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:46:53 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Code

def initialize(M:object)->None:
    
    dq = grad(M.H,argnums=1)
    dp = lambda q,p: -gradx(M.H)(q,p)
    
    def ode_Hamiltonian(c:tuple[ndarray, ndarray, ndarray],
                        y:ndarray
                        )->ndarray:
        t,x,chart = c
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])
        return jnp.stack((dqt,dpt))
    
    def chart_update_Hamiltonian(xp:ndarray,
                                 chart:ndarray,
                                 y:ndarray
                                 )->tuple[ndarray, ndarray]:
        if M.do_chart_update is None:
            return (xp,chart)
    
        p = xp[1]
        x = (xp[0],chart)
    
        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                            jnp.stack((new_x,M.update_covector(x,new_x,new_chart,p))),
                            xp),
                jnp.where(update,
                            new_chart,
                            chart))
    
    M.Hamiltonian_dynamics = jit(lambda q,p,dts: integrate(ode_Hamiltonian,chart_update_Hamiltonian,
                                                           jnp.stack((q[0] if type(q)==type(()) else q,p)),
                                                           q[1] if type(q)==type(()) else None,dts))
    
    def Exp_Hamiltonian(q:ndarray,p:ndarray,T:float=T,n_steps:int=n_steps)->tuple[ndarray,ndarray]:

        curve = M.Hamiltonian_dynamics(q,p,dts(T,n_steps))
        q = curve[1][-1,0]
        chart = curve[2][-1]

        return(q,chart)
    
    def Exp_Hamiltoniant(q:ndarray,p:ndarray,T:float=T,n_steps:int=n_steps)->tuple[ndarray, ndarray]:

        curve = M.Hamiltonian_dynamics(q,p,dts(T,n_steps))
        qs = curve[1][:,0]
        charts = curve[2]
        
        return(qs,charts)
    
    M.Exp_Hamiltonian = Exp_Hamiltonian
    M.Exp_Hamiltoniant = Exp_Hamiltoniant
    
    return

