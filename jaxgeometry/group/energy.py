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

#%% Energy

def energy(G:object)->None:
    """ group Lagrangian and Hamiltonian from invariant metric """

    def Lagrangian(g:Array,
                   vg:Array
                   )->Array:
        
        """ Lagrangian """
        
        return .5*G.gG(g,vg,vg)
    
    def Lagrangianpsi(q:Array,
                      v:Array
                      )->Array:
        
        """ Lagrangian using psi map """
        
        return .5*G.gpsi(q,v,v)
    
    def l(hatxi:Array)->Array:
        
        """ LA restricted Lagrangian """
        
        return 0.5*G.gV(hatxi,hatxi)

    def Hpsi(q:Array,
             p:Array
             )->Array:
        
        """ Hamiltonian using psi map """
        
        return .5*G.cogpsi(q,p,p)
    
    def Hminus(mu:Array)->Array:
        
        """ LA^* restricted Hamiltonian """
        
        return .5*G.cogV(mu,mu)

    def HL(q:Array,
           p:Array
           )->Array:
        
        """ Legendre transformation. The above Lagrangian is hyperregular """
        
        (q,v) = invFLpsi(q,p)
        
        return jnp.dot(p,v)-L(q,v)
    
    def hl(mu:Array)->Array:
        
        hatxi = invFl(mu)
        
        return jnp.dot(mu,hatxi)-l(hatxi)
    
    G.Lagrangian = Lagrangian
    
    G.Lagrangianpsi = Lagrangianpsi
    G.dLagrangianpsidq = jax.grad(G.Lagrangianpsi)
    G.dLagrangianpsidv = jax.grad(G.Lagrangianpsi)
    
    G.l = l
    G.dldhatxi = jax.grad(G.l)
    
    G.Hpsi = Hpsi
    
    G.Hminus = Hminus
    G.dHminusdmu = jax.grad(G.Hminus)
    
    G.FLpsi = lambda q,v: (q,G.dLagrangianpsidv(q,v))
    G.invFLpsi = lambda q,p: (q,G.cogpsi(q,p))
    
    G.HL = HL
    G.Fl = lambda hatxi: G.dldhatxi(hatxi)
    G.invFl = lambda mu: G.cogV(mu)
    
    G.hl = hl

    # default Hamiltonian
    G.H = lambda q,p: G.Hpsi(q[0],p) if type(q) == type(()) else G.Hpsi(q,p)
    
    return

# A.set_value(np.diag([3,2,1]))
# print(FLpsif(q0,v0))
# print(invFLpsif(q0,p0))
# (flq0,flv0)=FLpsif(q0,v0)
# print(q0,v0)
# print(invFLpsif(flq0,flv0))