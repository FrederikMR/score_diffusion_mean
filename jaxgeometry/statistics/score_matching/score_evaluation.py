#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:16:12 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *
from jaxgeometry.autodiff import jacfwdx

#%% Code

class ScoreEvaluation(object):
    def __init__(self,
                 M:object,
                 s1_model:object,
                 s1_state:dict,
                 s2_model:object=None,
                 s2_state:dict = None,
                 s2_approx:bool=True,
                 method:str='Local',
                 seed:int = 2712,
                 )->None:
        
        self.M = M
        self.s1_model = s1_model
        self.s1_state = s1_state
        self.s2_model = s2_model
        self.s2_state = s2_state
        self.s2_approx = s2_approx
        self.method = method
        self.rng_key = jrandom.PRNGKey(seed)
        
    def get_coords(self, Fx:Array)->Tuple[Array, Array]:
        
        chart = self.M.centered_chart(Fx)
        return (self.M.invF((Fx,chart)),chart)
    
    def grad_TM(self, Fx:Array, v:Array)->Array:
        
        x = self.get_coords(Fx)

        return jnp.dot(self.M.invJF((Fx,x[1])),v)
    
    def hess_TM(self, Fx:Array, v:Array, h:Array)->Array:
        
        x = self.get_coords(Fx)
        
        val1 = jacfwdx(lambda x: self.M.invJF((self.M.F(x), self.M.F(x))))(x)
        val2 = self.M.invJF((self.M.F(x), self.M.F(x)))
        val3 = self.M.JF(x)
        
        term1 = jnp.einsum('ijk,j->ik', val1, v)
        term2 = val2.dot(h).dot(val3)
        
        return term1+term2
    
    def grady_log(self, 
                  x:Tuple[Array, Array], 
                  y:Tuple[Array, Array], 
                  t:Array,
                  )->Array:
        
        if self.method == 'Embedded':
            return self.grad_TM(self.M.F(y), 
                                self.s1_model.apply(self.s1_state.params,self.rng_key, 
                                                    jnp.hstack((self.M.F(x), self.M.F(y), t))))
        else:
            return self.s1_model.apply(self.s1_state.params,self.rng_key, jnp.hstack((x[0], y[0], t)))
        
    def ggrady_log(self,
                   x:Tuple[Array, Array],
                   y:Tuple[Array, Array],
                   t:Array
                   )->Array:
        
        if self.method == 'Embedded':
            if self.s2_approx:
                s2 = self.s2_model.apply(self.s2_state.params,self.rng_key, jnp.hstack((x[1], y[1], t)))
            else:
                s2 = jacfwdx(lambda y: jnp.dot(self.M.invJF((self.M.F(y), y[1])), 
                                               self.s1_model.apply(self.s1_state.params,
                                                                   self.rng_key, 
                                                                   jnp.hstack((self.M.F(x), self.M.F(y), t)))))(y)
            s1 = self.s1_model.apply(self.s1_state.params,self.rng_key, 
                                jnp.hstack((self.M.F(x), self.M.F(y), t)))
            return s2#self.hess_TM(self.M.F(y), s1, s2)
        else:
            if self.s2_approx:
                s2 = self.s2_model.apply(self.s2_state.params, self.rng_key, jnp.hstack((x[0],y[0],t)))
            else:
                s2 = jacfwdx(lambda y: self.s1_model.apply(self.s1_state.params,
                                                           self.rng_key, 
                                                           jnp.hstack((x[0], y[0], t))))(y)
            return s2
        
    def gradt_log(self, 
                  x:Array,
                  y:Array,
                  t:Array, 
                  )->Array:

        s1_val = self.grady_log(x,y,t)
        s2_val = self.ggrady_log(x,y,t)

        div = jnp.trace(s2_val)+.5*jnp.dot(s1_val,jacfwdx(self.M.logAbsDet)(y).squeeze())

        return 0.5*(jnp.dot(s1_val, s1_val)+div)