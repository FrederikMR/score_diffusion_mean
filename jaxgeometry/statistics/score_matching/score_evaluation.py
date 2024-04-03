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
    
    def grad_TM(self, x:Array, v:Array)->Array:

        Jf = self.M.JF(x)
        #return jnp.dot(self.M.invJF((Fx,x[1])),v)
        return jnp.einsum('ij,i->j', Jf, v)
    
    def hess_TM(self, x:Array, v:Array, h:Array)->Array:
        #val1 = jacfwd(lambda x1: self.M.invJF((x1, x1)))(self.M.F(x))
        #val2 = self.M.invJF((x[1], self.M.F(x)))
        #val3 = self.M.JF(x)
        
        #term1 = jnp.einsum('ikj,jl->ikl', val1, val3)
        #term1 = jnp.einsum('ikl,k->il', term1, v)
        #term1 = jnp.einsum('ijk,j->ik', val1, v)
        #term2 = val2.dot(h).dot(val3)
        val1 = self.M.JF(x)
        val2 = jacfwdx(lambda x1: self.M.JF(x1))(x)
        term1 = jnp.einsum('jl,li,jk->ik', h, val1, val1)
        term2 = jnp.einsum('j,jik', v, val2)
        
        return term1+term2
    
    def hess_EmbeddedTM(self, Fx:Array, v:Array, h:Array)->Array:
        
        val1 = self.M.proj(Fx, h)
        val2 = v-self.M.proj(Fx, v)
        val3 = jacfwd(lambda x: self.M.proj(Fx, val2))(Fx)
        
        return val1+val3
    
    def grady_eval(self, 
                  x:Tuple[Array, Array], 
                  y:Tuple[Array, Array], 
                  t:Array,
                  )->Array:
        
        if self.s1_state == None:
            return self.s1_model(x,y,t)
        else:
            if self.method == 'Embedded':
                return self.s1_model.apply(self.s1_state.params,self.rng_key, jnp.hstack((self.M.F(x), self.M.F(y), t)))
            else:
                return self.s1_model.apply(self.s1_state.params,self.rng_key, jnp.hstack((x[0], y[0], t)))
        
    def grady_proj(self, 
                  x:Tuple[Array, Array], 
                  y:Tuple[Array, Array], 
                  t:Array,
                  )->Array:
        
        if self.method == 'Embedded':
            s1 = self.grady_eval(x,y,t)
            return self.M.proj(y,s1)
        else:
            return self.grady_eval(x,y,t)
        
    def ggrady_eval(self, 
                  x:Tuple[Array, Array], 
                  y:Tuple[Array, Array], 
                  t:Array,
                  )->Array:
        
        if self.s2_state == None:
            return self.s2_model(x,y,t)
        else:
            if self.s2_approx:
                if self.method == 'Embedded':
                    #s1 = self.grady_eval(x,y,t)
                    s2 = self.s2_model.apply(self.s2_state.params,self.rng_key, jnp.hstack((self.M.F(x), self.M.F(y), t)))
                    #s2 = self.hess_EmbeddedTM(y[1], s1, s2)
                    return s2
                else:
                    return self.s2_model.apply(self.s2_state.params,self.rng_key, jnp.hstack((x[0], y[0], t)))
            else:
                if self.method == 'Embedded':
                    return jacfwd(lambda Fy: \
                                  self.s1_model.apply(self.s1_state.params,self.rng_key, 
                                                      jnp.hstack((self.M.F(x), Fy, t))))(self.M.F(y))
                else:
                    x = x[0]
                    y = y[0]
                    return jacfwd(lambda y: \
                                  self.s1_model.apply(self.s1_state.params,self.rng_key, jnp.hstack((x, y, t))))(y)
    
    def grady_val(self, 
                  x:Tuple[Array, Array], 
                  y:Tuple[Array, Array], 
                  t:Array,
                  )->Array:
        
        if self.method == 'Embedded':
            s1 = self.M.proj(self.M.F(y),self.grady_eval(x,y,t))
            return self.grad_TM(y,s1)
        else:
            return self.grady_eval(x,y,t)
        
    def grady_log(self, 
                  x:Tuple[Array, Array], 
                  y:Tuple[Array, Array], 
                  t:Array,
                  )->Array:
        
        if self.method == 'Embedded':
            s1 = self.grad_TM(y, self.grady_eval(x,y,t))
        else:
            s1 = self.grady_eval(x,y,t)

        return s1#jnp.linalg.solve(self.M.g(y), s1)#jnp.dot(self.M.gsharp(y), s1)
        
    def ggrady_log(self,
                   x:Tuple[Array, Array],
                   y:Tuple[Array, Array],
                   t:Array
                   )->Array:
        if self.method == 'Embedded':
            if self.s2_approx:
                s2 = self.ggrady_eval(x,y,t)
                s1 = self.grady_eval(x,y,t)
                #s1 = self.M.proj(y[1],s1)
                #s2 = self.hess_EmbeddedTM(y[1], s1, s2)
                s2 = self.hess_TM(y, s1, s2)
                return s2
            else:
                s2 = jacfwdx(lambda y: jnp.dot(self.M.invJF((self.M.F(y), self.M.F(y))), self.grady_eval(x,y,t)))(y)
                return s2

            #(s2-jnp.einsum('mij,m->ij',self.M.Gamma_g(x),self.grady_val(x,y,t)))
        else:
            if self.s2_approx:
                s2 = self.ggrady_eval(x,y,t)
            else:
                s2 = jacfwdx(lambda y: self.grady_eval(x,y,t))(y)
    
            return s2#(s2-jnp.einsum('mij,m->ij',self.M.Gamma_g(y), self.grady_val(x,y,t)))
        
    def grad_debugging_log(self, 
                      x:Array,
                      y:Array,
                      t:Array, 
                      )->Array:
        
        s1_val = self.grady_log(x,y,t)
        s2_val = self.ggrady_log(x,y,t)
        s2_val = jnp.linalg.solve(self.M.g(y), s2_val)
        div = jnp.trace(s2_val)+.5*jnp.dot(s1_val,jacfwdx(self.M.logAbsDet)(x).squeeze())

        return s1_val, s2_val, div, .5*jnp.dot(s1_val,jacfwdx(self.M.logAbsDet)(x).squeeze()), jnp.dot(s1_val, s1_val), self.ggrady_log(x,y,t)
    
    def laplace_beltrami(self,
                         x:Array,
                         y:Array,
                         t:Array
                         )->Array:
        
        s1_val = self.grady_val(x,y,t)
        s2_val = self.ggrady_log(x,y,t)
        
        return jnp.trace(s2_val)+.5*jnp.dot(s1_val,jacfwdx(self.M.logAbsDet)(y).squeeze())
        
    def gradt_log(self, 
                  x:Array,
                  y:Array,
                  t:Array, 
                  )->Array:
        
        #s1_val = self.grady_val(x,y,t)
        #s1_val = self.grady_eval(x,y,t)
        #s2_val = self.ggrady_log(x,y,t)
        
        #div = jnp.trace(s2_val)
        
        #return 0.5*(jnp.dot(s1_val, s1_val)+div)

        s1_val = self.grady_val(x,y,t)
        #s1_val = self.grady_eval(x,y,t)
        s2_val = self.ggrady_log(x,y,t)
        #s2_val = self.ggrady_eval(x,y,t)
        #s2_val = jnp.linalg.solve(self.M.g(y), s2_val)
        #gamma= self.M.Gamma_g(y)
        #ginv = self.M.gsharp(y)
        #div = jnp.einsum('jk,jk->', ginv, s2_val)-jnp.einsum('jk,ljk,l->', ginv, gamma, s1_val)
        div = jnp.trace(s2_val)+.5*jnp.dot(s1_val,jacfwdx(self.M.logAbsDet)(y).squeeze())
        
        return 0.5*(jnp.dot(s1_val, s1_val)+div)