#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:54:42 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Iterative Maximum Likelihood Estimation

def iterative_mle(obss:ndarray,
                  neg_log_p_Ts:Callable,
                  params:tuple[...],
                  params_inds:tuple[...],
                  params_update:Callable,
                  chart:ndarray,
                  _dts:ndarray,
                  M:object,
                  N:int=1,
                  step_size:float=1e-1,
                  num_steps:int=50
                  )->tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:

    def step(step, params, opt_state, chart):
        
        params = get_params(opt_state)
        value,grads = vg(params[0],chart,obss,dWs(len(obss[0])*N*M.dim,_dts).reshape(-1,_dts.shape[0],N,M.dim),_dts,*params[1:])
        opt_state = opt_update(step, grads, opt_state)
        opt_state,chart = params_update(opt_state, chart)
        
        return (value,opt_state,chart)
    
    opt_init, opt_update, get_params = optimizers.adam(step_size)
    vg = jax.value_and_grad(neg_log_p_Ts,params_inds)

    opt_state = opt_init(params)
    values = (); paramss = ()

    for i in range(num_steps):
        (value, opt_state, chart) = step(i, params, opt_state, chart)
        values += (value,); paramss += ((*get_params(opt_state),chart),)
        if i % 1 == 0:
            print("Step {} | T: {:0.6e} | T: {}".format(i, value, str((get_params(opt_state),chart))))
    print("Final {} | T: {:0.6e} | T: {}".format(i, value, str(get_params(opt_state))))
    
    return (get_params(opt_state),chart,value,jnp.array(values),paramss)