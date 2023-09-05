#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:43:31 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Code

def initialize(M:object)->None:
    """ flow along a vector field X """
    def flow(X:Callable[[ndarray], ndarray]):

        def ode_flow(c:tuple[ndarray, ndarray, ndarray],
                     y:ndarray
                     )->ndarray:
            
            t,x,chart = c
            
            return X((x,chart))
        
        def chart_update_flow(x:ndarray,chart:ndarray,*ys)->ndarray:
            if M.do_chart_update is None:
                return (x,chart)

            update = M.do_chart_update(x)
            new_chart = M.centered_chart((x,chart))
            new_x = M.update_coords((x,chart),new_chart)[0]

            return (jnp.where(update,
                                    new_x,
                                    x),
                    jnp.where(update,
                                    new_chart,
                                    chart),
                    )
        
        flow = jit(lambda x,dts: integrate(ode_flow,chart_update_flow,x[0],x[1],dts))
        return flow
    
    M.flow = flow
    
    return