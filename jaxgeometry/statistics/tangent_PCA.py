#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:44:19 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Tanget Space PCA

## THIS NEED TO BE UPDATED TO NOT USE mpu, WHICH IS OUTDATED WHEN USING JAX

def tangent_PCA(M:object, 
                Log:Callable, 
                mean:ndarray, 
                y:tuple[...]
                )->ndarray:
    y = list(y) # make sure y is subscriptable
    
    try:
        mpu.openPool()
        N = len(y)
        sol = mpu.pool.imap(lambda pars: (Log(mean,y[pars[0]],)[0],),mpu.inputArgs(range(N)))
        res = list(sol)
        Logs = mpu.getRes(res,0)
    except:
        mpu.closePool()
        raise
    else:
        mpu.closePool()

    pca = PCA()
    pca.fit(Logs)
    pca.transformed_Logs = pca.transform(Logs)

    return pca