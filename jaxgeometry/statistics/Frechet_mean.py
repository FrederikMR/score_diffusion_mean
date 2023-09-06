#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:37:03 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Fréchet mean

def initialize(M:object,
               Exp:Callable[[tuple[ndarray, ndarray]], tuple[ndarray, ndarray]]=None
               )->None:

    # objective
    def f(chart:ndarray,x:ndarray,v:ndarray)->ndarray:
        
        return jnp.dot(v,jnp.dot(M.g((x,chart)),v))
    
    # constraint
    def _c(chart:ndarray,x:ndarray,v:ndarray,y:ndarray,ychart:ndarray)->ndarray:
        
        xT,chartT = M.Exp((x,chart),v)
        y_chartT = M.update_coords((y,ychart),chartT)
        
        return jnp.sqrt(M.dim)*(xT-y_chartT[0])
    
    def c(chart:ndarray,x:ndarray,v:ndarray,y:ndarray,ychart:ndarray)->ndarray:
        
        return jnp.sum(jnp.square(_c(chart,x,v,y,ychart)))
    
    def vgx_f(chart:ndarray,
              x:ndarray,
              v:ndarray,
              y:ndarray,
              ychart:ndarray)->tuple[ndarray, ndarray]:
        
        _jacxv_c = M.Frechet_mean_jacxv_c(chart,x,v,y,ychart)
        jacv = -jnp.linalg.solve(_jacxv_c[1],_jacxv_c[0]) # implicit function theorem
    
        v_f, g_f = M.Frechet_mean_jacxv_f(chart,x,v)
        g_f = g_f[0]+jnp.dot(g_f[1],jacv)
    
        return v_f, g_f

    def Frechet_mean(ys:tuple[...],
                     x0:tuple[ndarray, ndarray],
                     Log:Callable=None,
                     options:dict={}
                     )->tuple[tuple[ndarray, ndarray], ndarray, int]:
        # data
        ys = list(ys) # make sure y is subscriptable
        N = len(ys)
        chart = x0[1] # single chart for now, could be updated

        if Log is None:
            # combined optimization, no use of Log maps
            step_sizex=options.get('step_sizex',1e-1)
            step_sizevs=options.get('step_sizevs',1e-1)
            num_steps=options.get('num_steps',200)
            optx_update_mod=options.get('optx_update_mod',5)
    
            opt_initx, opt_updatex, get_paramsx = optimizers.adam(step_sizex)
            opt_initvs, opt_updatevs, get_paramsvs = optimizers.adam(step_sizevs)

            # tracking steps
            steps = (x0,)
        
            def step(step, params, ys, y_charts, opt_statex, opt_statevs):
                paramsx = get_paramsx(opt_statex); paramsvs = get_paramsvs(opt_statevs)
                valuex = None; gradx = jnp.zeros(M.dim); valuevs = (); gradvs = ()
        #         for i in range(N):
        #             vvs,gvs = vgv_c(params,paramsx,paramsvs[i],*ys[i])
        #             valuevs += (vvs,); gradvs += gvs
                
                valuevs,gradvs = jax.vmap(M.Frechet_mean_vgv_c,(None,None,0,0,0))(params,paramsx,paramsvs,ys,y_charts)
                opt_statevs = opt_updatevs(step, jnp.array(gradvs).squeeze(), opt_statevs)
                if step % optx_update_mod == 0:
        #             for i in range(N):
        #                 vx,gx = vgx_f(params,paramsx,paramsvs[i],*ys[i])
        #                 valuex += 1/N*vx; gradx = gradx+1/N*gx[0]
                    valuex,gradx = jax.vmap(M.Frechet_mean_vgx_f,(None,None,0,0,0))(params,paramsx,paramsvs,ys,y_charts)
                    valuex = jnp.mean(valuex,0); gradx = jnp.mean(gradx,0)
                    opt_statex = opt_updatex(step, gradx, opt_statex)
                return (valuex, valuevs), (opt_statex, opt_statevs)
        
            # optim setup
            params = x0[1]
            paramsx = x0[0]
            paramsvs = jnp.zeros((N,M.dim))
            opt_statex = opt_initx(paramsx)
            opt_statevs = opt_initvs(paramsvs)
            valuex = 0; valuesvs = ()
            ys,y_charts=list(zip(*ys))
            ys = jnp.array(ys); y_charts = jnp.array(y_charts)
        
            for i in range(num_steps):
                (_valuex, valuevs), (opt_statex, opt_statevs) = step(i, params, ys, y_charts, opt_statex, opt_statevs)
                if _valuex:
                    valuex = _valuex
                if i % 10 == 0:
                    print("Step {} | T: {:0.6e} | T: {:0.6e}".format(i, valuex, jnp.max(valuevs)))
                if i % optx_update_mod == 0:
                    steps += ((get_paramsx(opt_statex),chart),)
            print("Step {} | T: {:0.6e} | T: {:0.6e} ".format(i, valuex, jnp.max(valuevs)))
        
            m = (get_paramsx(opt_statex),params)
            vs = get_paramsvs(opt_statevs)
        
            return (m,valuex,steps,vs)
            
    
        else:
            # Log based optimization
            def fopts(x):
                N = len(ys)
                Logs = np.zeros((N, x.shape[0]))
                for i in range(N):
                    Logs[i] = Log((x,chart), ys[i])[0]
    
                res = (1. / N) * np.sum(np.square(Logs))
                grad = -(2. / N) * np.sum(Logs, 0)
    
                return (res, grad)
    
            # tracking steps
            global _steps
            _steps = (x0,)
            def save_step(k):
                global _steps
                _steps += ((k,chart),)
    
            res = minimize(fopts, x0[0], method='BFGS', jac=True, options=options, callback=save_step)
    
            return ((res.x,x0[1]), res.fun, _steps)
        
    try:
        if Exp is None:
            Exp = M.Exp
    except AttributeError:
        return
    
    M.Frechet_mean_f = f
    
    # derivatives
    M.Frechet_mean_vgv_c = jit(jax.value_and_grad(c,(2,)))
    M.Frechet_mean_jacxv_c = jit(jax.jacrev(_c,(1,2)))
    M.Frechet_mean_jacxv_f = jit(jax.value_and_grad(f,(1,2)))
    
    M.Frechet_mean_vgx_f = vgx_f

    M.Frechet_mean = Frechet_mean
    
    return