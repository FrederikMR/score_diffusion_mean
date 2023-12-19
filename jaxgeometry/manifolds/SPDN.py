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

#https://indico.ictp.it/event/a08167/session/124/contribution/85/material/0/0.pdf
#https://www.cis.upenn.edu/~cis6100/geomean.pdf
#https://poisson.phc.dm.unipi.it/~maxreen/bruno/pdf/D.%20Bini%20and%20B.%20Iannazzo%20-%20A%20note%20on%20computing%20Matrix%20Geometric%20Means.pdf
#https://manoptjl.org/v0.1/manifolds/hyperbolic/
#https://proceedings.neurips.cc/paper_files/paper/2020/file/1aa3d9c6ce672447e1e5d0f1b5207e85-Paper.pdf

#%% Modules

from jaxgeometry.setup import *
from .riemannian import EmbeddedManifold, metric, curvature, geodesic, Log, parallel_transport

#%% Symmetric Positive Definite Space in Local Coordinates

class SPDN(EmbeddedManifold):
    """ manifold of symmetric positive definite matrices """

    def __init__(self,N=3):
        self.N = N
        dim = N*(N+1)//2
        emb_dim = N*N
        EmbeddedManifold.__init__(self,
                                  F=lambda x: F(self, x),
                                  dim=dim,
                                  emb_dim=emb_dim, 
                                  invF=lambda x: invF(self, x))

        self.act = lambda g,q: jnp.tensordot(g,jnp.tensordot(q.reshape((N,N)),g,(1,1)),(1,0)).flatten()
        self.acts = vmap(self.act,(0,None))
        
        self.Dn = DupMat(N)
        self.Dp = invDupMat(N)
        self.g = lambda x: g(self, x)
        self.do_chart_update = lambda x: False
        
        metric(self)
        curvature(self)
        geodesic(self)
        Log(self)
        parallel_transport(self)
        
        self.gsharp = lambda x: gsharp(self,x)
        self.det = lambda x,A=None: det(self, x, A)
        self.Gamma_g = lambda x: Gamma(self, x)
        self.Expt = lambda x,v,T: Expt(self, x, v, T)
        self.Exp = lambda x,v: self.Expt(x,v,T=1.0)
        self.Log = lambda x,y: Log(self, x, y)
        self.dist = lambda x,y: dist(self, x,y)
        self.dot = lambda x,v,w: dot(self, x, v, w)
        self.ProjTM = lambda x,v: ProjTM(self, x, v)
        self.ParallelTransport = lambda x,y,v: ParallelTransport(self, x, y, v)
        self.DupMat = DupMat
        self.invDupMat = invDupMat

    def __str__(self):
        return "SPDN(%d), dim %d" % (self.N,self.dim)
    
    def centered_chart(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return stop_gradient(self.F(x))
        else:
            return x#self.F((x,jnp.zeros(self.N*self.N))) # already in embedding space
    
    def chart(self):
        """ return default coordinate chart """
        return jnp.eye(self.N).reshape(-1)

    def plot(self, rotate=None, alpha = None):
        ax = plt.gca()
        #ax.set_aspect("equal")
        if rotate != None:
            ax.view_init(rotate[0],rotate[1])
    #     else:
    #         ax.view_init(35,225)
        plt.xlabel('x')
        plt.ylabel('y')


    def plot_path(self, x,color_intensity=1.,color=None,linewidth=3.,prevx=None,ellipsoid=None,i=None,maxi=None):
        assert(len(x.shape)>1)
        for i in range(x.shape[0]):
            self.plotx(x[i],
                  linewidth=linewidth if i==0 or i==x.shape[0]-1 else .3,
                  color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                  prevx=x[i-1] if i>0 else None,ellipsoid=ellipsoid,i=i,maxi=x.shape[0])
        return

    def plotx(self, x,color_intensity=1.,color=None,linewidth=3.,prevx=None,ellipsoid=None,i=None,maxi=None):
        x = x.reshape((self.N,self.N))
        (w,V) = np.linalg.eigh(x)
        s = np.sqrt(w[np.newaxis,:])*V # scaled eigenvectors
        if prevx is not None:
            prevx = prevx.reshape((self.N,self.N))
            (prevw,prevV) = np.linalg.eigh(prevx)
            prevs = np.sqrt(prevw[np.newaxis,:])*prevV # scaled eigenvectors
            ss = np.stack((prevs,s))

        colors = color_intensity*np.array([[1,0,0],[0,1,0],[0,0,1]])
        if ellipsoid is None:
            for i in range(s.shape[1]):
                plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)
                if prevx is not None:
                    plt.plot(ss[:,0,i],ss[:,1,i],ss[:,2,i],linewidth=.3,color=colors[i])
        else:
            try:
                if i % int(ellipsoid['step']) != 0 and i != maxi-1:
                    return
            except:
                pass
            try:
                if ellipsoid['subplot']:
                    (fig,ax) = newfig3d(1,maxi//int(ellipsoid['step'])+1,i//int(ellipsoid['step'])+1,new_figure=i==0)
            except:
                (fig,ax) = newfig3d()
            #draw ellipsoid, from https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
            U, ss, rotation = np.linalg.svd(x)  
            radii = np.sqrt(ss)
            u = np.linspace(0., 2.*np.pi, 20)
            v = np.linspace(0., np.pi, 10)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for l in range(x.shape[0]):
                for k in range(x.shape[1]):
                    [x[l,k],y[l,k],z[l,k]] = np.dot([x[l,k],y[l,k],z[l,k]], rotation)
            ax.plot_surface(x, y, z, facecolors=cm.winter(y/np.amax(y)), linewidth=0, alpha=ellipsoid['alpha'])
            for i in range(s.shape[1]):
                plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)
            plt.axis('off')
            
#%% Symmetric Positive Definite Space in Embedded Space
            
class SPDN_Ambient(EmbeddedManifold):
    """ manifold of symmetric positive definite matrices """

    def __init__(self,N=3):
        self.N = N
        dim = N*(N+1)//2
        emb_dim = N*N
        EmbeddedManifold.__init__(self,F=lambda x: F(self, x),
                                      dim=dim,
                                      emb_dim=emb_dim, 
                                      invF=lambda x: invF(self, x))

        self.act = lambda g,q: jnp.tensordot(g,jnp.tensordot(q.reshape((N,N)),g,(1,1)),(1,0)).flatten()
        self.acts = vmap(self.act,(0,None))
        
        self.Dn = DupMat(N)
        self.Dp = invDupMat(N)
        self.g = lambda x: g(self, x)
        self.do_chart_update = lambda x: False
        
        riemannian.metric(self)
        riemannian.curvature(self)
        riemannian.geodesic(self)
        riemannian.Log(self)
        riemannian.parallel_transport(self)
        
        self.gsharp = lambda x: gsharp(self,x)
        self.det = lambda x,A=None: det(self, x, A)
        self.Gamma_g = lambda x: Gamma(self, x)
        self.Expt = lambda x,v,T: Expt(self, x, v, T)
        self.Exp = lambda x,v: self.Expt(x,v,T=1.0)
        self.Log = lambda x,y: Log(self, x, y)
        self.dist = lambda x,y: dist(self, x,y)
        self.dot = lambda x,v,w: dot(self, x, v, w)
        self.ProjTM = lambda x,v: ProjTM(self, x, v)
        self.ParallelTransport = lambda x,y,v: ParallelTransport(self, x, y, v)
        self.DupMat = DupMat
        self.invDupMat = invDupMat

    def __str__(self):
        return "SPDN(%d), dim %d" % (self.N,self.dim)
    
    def centered_chart(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return stop_gradient(self.F(x))
        else:
            return x#self.F((x,jnp.zeros(self.N*self.N))) # already in embedding space
    
    def chart(self):
        """ return default coordinate chart """
        return jnp.eye(self.N).reshape(-1)

    def plot(self, rotate=None, alpha = None):
        ax = plt.gca()
        #ax.set_aspect("equal")
        if rotate != None:
            ax.view_init(rotate[0],rotate[1])
    #     else:
    #         ax.view_init(35,225)
        plt.xlabel('x')
        plt.ylabel('y')


    def plot_path(self, x,color_intensity=1.,color=None,linewidth=3.,prevx=None,ellipsoid=None,i=None,maxi=None):
        assert(len(x.shape)>1)
        for i in range(x.shape[0]):
            self.plotx(x[i],
                  linewidth=linewidth if i==0 or i==x.shape[0]-1 else .3,
                  color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                  prevx=x[i-1] if i>0 else None,ellipsoid=ellipsoid,i=i,maxi=x.shape[0])
        return

    def plotx(self, x,color_intensity=1.,color=None,linewidth=3.,prevx=None,ellipsoid=None,i=None,maxi=None):
        x = x.reshape((self.N,self.N))
        (w,V) = np.linalg.eigh(x)
        s = np.sqrt(w[np.newaxis,:])*V # scaled eigenvectors
        if prevx is not None:
            prevx = prevx.reshape((self.N,self.N))
            (prevw,prevV) = np.linalg.eigh(prevx)
            prevs = np.sqrt(prevw[np.newaxis,:])*prevV # scaled eigenvectors
            ss = np.stack((prevs,s))

        colors = color_intensity*np.array([[1,0,0],[0,1,0],[0,0,1]])
        if ellipsoid is None:
            for i in range(s.shape[1]):
                plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)
                if prevx is not None:
                    plt.plot(ss[:,0,i],ss[:,1,i],ss[:,2,i],linewidth=.3,color=colors[i])
        else:
            try:
                if i % int(ellipsoid['step']) != 0 and i != maxi-1:
                    return
            except:
                pass
            try:
                if ellipsoid['subplot']:
                    (fig,ax) = newfig3d(1,maxi//int(ellipsoid['step'])+1,i//int(ellipsoid['step'])+1,new_figure=i==0)
            except:
                (fig,ax) = newfig3d()
            #draw ellipsoid, from https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
            U, ss, rotation = np.linalg.svd(x)  
            radii = np.sqrt(ss)
            u = np.linspace(0., 2.*np.pi, 20)
            v = np.linspace(0., np.pi, 10)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for l in range(x.shape[0]):
                for k in range(x.shape[1]):
                    [x[l,k],y[l,k],z[l,k]] = np.dot([x[l,k],y[l,k],z[l,k]], rotation)
            ax.plot_surface(x, y, z, facecolors=cm.winter(y/np.amax(y)), linewidth=0, alpha=ellipsoid['alpha'])
            for i in range(s.shape[1]):
                plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)
            plt.axis('off')
            
#%% Embedding

def F(M:object, x:Tuple[Array, Array])->Array:
    
    l = jnp.zeros((M.N, M.N))
    l = l.at[jnp.triu_indices(M.N, k=0)].set(x[0])
    
    return l.T.dot(l).reshape(-1)

def invF(M:object, x:Tuple[Array, Array])->Array:
    
    P = x[0].reshape(M.N, M.N)
    l = jnp.linalg.cholesky(P).T
    
    l = l[jnp.triu_indices(M.N, k=0)]  
    
    return l.reshape(-1)
            
#%% Metric

def g(N, x:Tuple[Array,Array])->Array:
    
    P = F(N,x).reshape(N, N)
    D = duplication_inv_fun(N)
    
    return jnp.matmul(D,jnp.linalg.solve(jnp.kron(P,P), D.T))

def gsharp(M:object, x:Tuple[Array,Array])->Array:
    
    P = M.F(x).reshape(M.N, M.N)
    D = M.dupmat_inv
    
    return jnp.matmul(D,jnp.matmul(jnp.kron(P,P), D.T))

def det(M:object, x:Tuple[Array, Array],A:Array=None)->Array: 
    
    P = M.F(x).reshape(M.N, M.N)
    
    return 2**((M.N*(M.N-1)//2))*jnp.linalg.det(P.reshape(M.N,M.N))**(M.N+1) if A is None \
        else jnp.linalg.det(jnp.tensordot(M.g(x),A,(1,0)))
        
def Gamma(M:object, x:Tuple[Array, Array])->Array: 
    
    p = M.N*(M.N+1)//2
    E = jnp.eye(M.N*M.N)[:p]
    D = M.dupmat
    P = M.F(x).reshape(M.N,M.N)
        
    return -vmap(lambda e: jnp.matmul(D.T,jnp.matmul(jnp.kron(jnp.linalg.inv(P), e.reshape(M.N,M.N)), D)))(E)

def Expt(M:object, x:Tuple[Array, Array], v:Array, t:float)->Array:
    
    P = M.F(x).reshape(M.N,M.N)
    v = M.ProjTM(x, v.reshape(M.N,M.N))
    
    P_phalf = jnp.array(jscipy.linalg.sqrtm(P), dtype=jnp.float32)
    P_nhalf = jnp.array(jscipy.linalg.sqrtm(jnp.linalg.inv(P)), dtype=jnp.float32)
    
    P_exp = jnp.matmul(jnp.matmul(P_phalf, \
                                 jscipy.linalg.expm(t*jnp.matmul(jnp.matmul(P_nhalf, v), P_nhalf))),
                      P_phalf)
    
    return (M.invF((x[1], P_exp)), P_exp.reshape(-1))

def ParallelTransport(M:object, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:
    
    P1 = M.F(x).reshape(M.N,M.N)
    P2 = M.F(y).reshape(M.N,M.N)
    v = M.ProjTM(x, v.reshape(M.N,M.N))
    
    P1_phalf = fractional_matrix_power(P1,0.5)
    P1_nahlf = fractional_matrix_power(P1,-0.5)
    logxy = M.Log(x,y)
    expxy = jscipy.linalg.expm(0.5*jnp.matmul(jnp.matmul(P1_nhalf, logxy), P1_nhalf))
    
    psi = jnp.matmul(jnp.matmul(P1_phalf, v), P1_phalf)
    term1 = jnp.matmul(P1_phalf, expxy)
    term2 = jnp.matmul(expxy, P1_phalf)
    
    return jnp.matmul(jnp.matmul(term1, psi), term2)
        
def Log(M:object, x:Tuple[Array, Array], y:Tuple[Array,Array])->Array:
    
    P = M.F(x).reshape(M.N,M.N)
    S = M.F(y).reshape(M.N,M.N)
    
    P_phalf = fractional_matrix_power(P,0.5)
    P_nhalf = fractional_matrix_power(P,-0.5)
    
    return jnp.matmul(jnp.matmul(P_phalf, \
                                 logm(jnp.matmul(jnp.matmul(P_nhalf, S), P_nhalf))),
                      P_phalf)

def dist(M:object, x:Tuple[Array, Array], y:Tuple[Array,Array])->Array:
    
    P1 = M.F(x)
    P2 = M.F(y)
    
    U, S, Vh = jnp.linalg.svd(jnp.linalg.solve(P1, P2))
    
    return jnp.sqrt(jnp.sum(jnp.log(S)**2))

def dot(M:object, x:Tuple[Array, Array], v:Array, w:Array)->Array:
    
    P = M.F(x)
    
    v1 = jnp.linalg.solve(P, v)
    v2 = jnp.linalg.solve(p, w)
    
    return jnp.trace(jnp.matmul(v1,v2))

def ProjTM(M:object, x:Tuple[Array, Array], v:Array) -> Array:
    
    P1_phalf = fractional_matrix_power(P, 0.5)
    v_symmetric = 0.5*(v+v.T)
    
    return jnp.matmul(jnp.matmul(P1_phalf, v_symmetric), P1_phalf)

#%% Duplication Matrix

def DupMat(N:int):
    
    def step_col(carry, i, j):
        
        D, A, u, i = carry
        A,u = 0.*A, 0.*u
        idx = j*N+i-((j+1)*j)//2
        
        A = A.at[i,i-j].set(1)
        A = A.at[i-j,i].set(1)
        u = u.at[idx].set(1)
        D += u.dot(A.reshape((1, -1), order="F"))
        i += 1
        
        return (D,A,u,i), None
        
    p = N*(N+1)//2
    A,D,u = jnp.zeros((N,N)), jnp.zeros((p,N*N)), jnp.zeros((p,1))    
    
    for j in range(N):
        D, _, _, _ = scan(lambda carry, i: step_col(carry, i, j), init=(D,A,u,j), xs=jnp.arange(0,N-j,1))[0]
    
    return D.T

def invDupMat(N:int):
    
    def step_col(carry, i, j, val):
        
        D, A, u, i = carry
        A,u = 0.*A, 0.*u
        idx = j*N+i-((j+1)*j)//2
        
        A = A.at[i,i-j].set(val)
        A = A.at[i-j,i].set(val)
        u = u.at[idx].set(1)
        D += u.dot(A.reshape((1, -1), order="F"))
        i += 1
        
        return (D,A,u,i), None
        
    p = N*(N+1)//2
    A,D,u = jnp.zeros((N,N)), jnp.zeros((p,N*N)), jnp.zeros((p,1))    
    
    D, _, _, _ = scan(lambda carry, i: step_col(carry, i, 0, 1.0), init=(D,A,u,0), xs=jnp.arange(0,N,1))[0]
    for j in range(1,N):
        D, _, _, _ = scan(lambda carry, i: step_col(carry, i, j, 0.5), init=(D,A,u,j), xs=jnp.arange(0,N-j,1))[0]
    
    return D

