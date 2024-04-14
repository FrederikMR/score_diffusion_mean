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
                 s1_model:Callable[[Array, Array, Array], Array],
                 s2_model:Callable[[Array, Array, Array], Array]=None,
                 method:str='Local',
                 )->None:
        
        if method not in ['Local', 'Embedded']:
            raise ValueError(f"Method is {method}. It should be either: Local, Embedded")
        
        self.M = M
        self.method = method
            
        if method == "Embedded":
            self.s1_model = lambda x,y,t: s1_model(self.M.F(x), self.M.F(y), t)
        else:
            self.s1_model = lambda x,y,t: s1_model(x[0],y[0],t)
        
        if s2_model is None:
            if method == "Embedded":
                self.s2_model = lambda x,y,t: jacfwd(lambda Fy: s1_model(self.M.F(x), Fy, t))(self.M.F(y))
            else:
                self.s2_model = lambda x,y,t: jacfwd(lambda y1: s1_model(x[0],y1,t))(y[0])
        else:
            if method == "Embedded":
                self.s2_model = lambda x,y,t: s2_model(self.M.F(x), self.M.F(y), t)
            else:
                self.s2_model = lambda x,y,t: s2_model(x[0], y[0], t)
        
    
    def update_coords(self, Fx:Array)->Tuple[Array, Array]:
        
        chart = self.M.centered_chart(Fx)
        
        return (self.M.invF((Fx,chart)),chart)
    
    def grad_local(self, x:Tuple[Array,Array], v:Array)->Array:
        
        Jf = self.M.JF(x)

        return jnp.einsum('ij,i->j', Jf, v)
    
    def grad_TM(self, Fx:Array, v:Array)->Array:

        return self.M.proj(Fx, v)
    
    def hess_local(self, x:Tuple[Array,Array], v:Array, h:Array)->Array:
        
        val1 = self.M.JF(x)
        val2 = jacfwdx(lambda x1: self.M.JF(x1))(x)
        term1 = jnp.einsum('jl,li,jk->ik', h, val1, val1)
        term2 = jnp.einsum('j,jik', v, val2)
        
        return term1+term2
    
    def hess_TM(self, Fx:Array, v:Array, h:Array)->Array:
        
        val1 = self.M.proj(Fx, h)
        val2 = v-self.M.proj(Fx, v)
        val3 = jacfwd(lambda x: self.M.proj(Fx, val2))(Fx)
        
        return val1+val3
    
    def grady_log(self, 
                  x:Tuple[Array, Array], 
                  y:Tuple[Array, Array], 
                  t:Array,
                  )->Array:
        
        if self.method == "Embedded":
            #x = self.update_coords(x)
            #y = self.update_coords(y)
            v = self.s1_model(x,y,t)
            return self.grad_local(y, v)
        else:
            s1 = self.s1_model(x,y,t)
            return s1
        
    def ggrady_log(self,
                   x:Tuple[Array, Array],
                   y:Tuple[Array, Array],
                   t:Array
                   )->Array:
        
        if self.method == "Embedded":
            #x = self.update_coords(x)
            #y = self.update_coords(y)
            h = self.s2_model(x,y,t)
            v = self.s1_model(x,y,t)
            return self.hess_local(y,v,h)
        else:
            s2 = self.s2_model(x,y,t)
            return s2
        
    def gradt_log(self, 
                  x:Tuple[Array,Array],
                  y:Tuple[Array,Array],
                  t:Array, 
                  )->Array:
        
        #if self.method == "Embedded":
            #x = self.update_coords(x)
            #y = self.update_coords(y)
        
        s1 = self.grady_log(x,y,t)
        s2 = self.ggrady_log(x,y,t)
        
        if self.method == "Embedded":
            laplace_beltrami = jnp.trace(s2)
            norm_s1 = jnp.dot(s1,s1)
        else:
            s1 = jnp.linalg.solve(self.M.g(y), s1)
            s2 = jnp.linalg.solve(self.M.g(y), s2)
            laplace_beltrami = jnp.trace(s2)+.5*jnp.dot(s1,jacfwdx(self.M.logAbsDet)(y).squeeze())
            norm_s1 = self.M.dot(y,s1,s1)
        
        return 0.5*(laplace_beltrami+norm_s1)