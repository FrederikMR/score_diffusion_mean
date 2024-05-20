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
                                  F=self.F,
                                  dim=dim,
                                  emb_dim=emb_dim, 
                                  invF=self.invF)

        self.act = lambda g,q: jnp.tensordot(g,jnp.tensordot(q.reshape((N,N)),g,(1,1)),(1,0)).flatten()
        self.acts = vmap(self.act,(0,None))
        
        self.Dn = self.DupMat(N)
        self.Dp = self.invDupMat(N)
        #self.g = self.Stdg
        self.do_chart_update = lambda x: False
        
        metric(self)
        curvature(self)
        geodesic(self)
        Log(self)
        parallel_transport(self)
        
        #self.gsharp = self.Stdgsharp
        #self.det = self.Stddet
        #self.Gamma_g = self.StdGamma
        #self.Expt = self.StdExpt
        #self.Exp = lambda x,v: self.Expt(x,v,t=1.0)
        self.ExpEmbedded = self.ExpEmbedded
        self.Log = self.StdLog
        self.dist = self.StdDist
        self.dot = self.StdDot
        self.proj = self.StdProj
        self.ParallelTransport = self.StdParallelTransport

    def __str__(self):
        return "SPDN(%d), dim %d" % (self.N,self.dim)
    
    def F(self, x:Tuple[Array, Array])->Array:
        
        l = jnp.zeros((self.N, self.N))
        l = l.at[jnp.triu_indices(self.N, k=0)].set(x[0])
        
        return l.T.dot(l).reshape(-1)
    
    def invF(self, x:Tuple[Array, Array])->Array:
        
        P = x[0].reshape(self.N, self.N)
        l = jnp.linalg.cholesky(P).T
        
        l = l[jnp.triu_indices(self.N, k=0)]  
        
        return l.reshape(-1)
    
    def Stdg(self, x:Tuple[Array,Array])->Array:
        
        P = self.F(x).reshape(self.N, self.N)
        D = self.Dp
        
        return jnp.matmul(D,jnp.linalg.solve(jnp.kron(P,P), D.T))
    
    def Stdgsharp(self, x:Tuple[Array,Array])->Array:
        
        P = self.F(x).reshape(self.N, self.N)
        D = self.Dn.T
        
        return jnp.matmul(D,jnp.matmul(jnp.kron(P,P), D.T))
    
    def Stddet(self, x:Tuple[Array, Array],A:Array=None)->Array: 
        
        P = self.F(x).reshape(self.N, self.N)
        
        return 2**((self.N*(self.N-1)//2))*jnp.linalg.det(P.reshape(self.N,self.N))**(self.N+1) if A is None \
            else jnp.linalg.det(jnp.tensordot(self.Stdg(x),A,(1,0)))
            
    def StdGamma(self, x:Tuple[Array, Array])->Array: 
        
        p = self.N*(self.N+1)//2
        E = jnp.eye(self.N*self.N)[:p]
        D = self.Dn
        P = self.F(x).reshape(self.N,self.N)
        pinv = jnp.linalg.inv(P)
            
        return -vmap(lambda e: jnp.matmul(D.T,jnp.matmul(jnp.kron(pinv, e.reshape(self.N,self.N)), D)))(E)
    
    def StdExpt(self, x:Tuple[Array, Array], v:Array, t:float=1.0)->Array:
        
        P = self.F(x).reshape(self.N,self.N)
        v = jnp.dot(self.JF(x),v).reshape(self.N,self.N)
        
        U,S,V = jnp.linalg.svd(P)
        P_phalf = jnp.dot(jnp.dot(U, jnp.diag(jnp.sqrt(S))), V)#jnp.linalg.cholesky(P)
        P_nhalf = jnp.linalg.inv(P_phalf)#jnp.linalg.inv(P_phalf)
        
        exp_val = jnp.dot(jnp.dot(P_nhalf, v), P_nhalf)
        exp_val = jscipy.linalg.expm(exp_val)
        
        P_exp = jnp.dot(jnp.dot(P_phalf, exp_val), P_phalf)
        P_exp = 0.5*(P_exp+P_exp.T)
        
        return (self.invF((P_exp, P_exp)), P_exp.reshape(-1))

    def ExpEmbedded(self, Fx:Array, v:Array, t:float=1.0)->Array:
        
        P = Fx.reshape(self.N,self.N)
        v = v.reshape(self.N,self.N)
        
        U,S,V = jnp.linalg.svd(P)
        P_phalf = jnp.dot(jnp.dot(U, jnp.diag(jnp.sqrt(S))), V)#jnp.linalg.cholesky(P)
        P_nhalf = jnp.linalg.inv(P_phalf)#jnp.linalg.inv(P_phalf)
        
        exp_val = jnp.dot(jnp.dot(P_nhalf, v), P_nhalf)
        exp_val = jscipy.linalg.expm(exp_val)
        
        P_exp = jnp.dot(jnp.dot(P_phalf, exp_val), P_phalf)
        P_exp = 0.5*(P_exp+P_exp.T) #For numerical stability
        
        return lax.select(jnp.linalg.det(P_exp)<1e-2  , P.reshape(-1), P_exp.reshape(-1))
    
    def StdLog(self, x:Tuple[Array, Array], y:Array)->Array:
        
        P = self.F(x).reshape(self.N,self.N)
        Q = y.reshape(self.N,self.N)
        
        U,S,V = jnp.linalg.svd(P)
        P_phalf = jnp.dot(jnp.dot(U, jnp.diag(jnp.sqrt(S))), V)#jnp.linalg.cholesky(P)
        P_nhalf = jnp.linalg.inv(P_phalf)#jnp.linalg.inv(P_phalf)
        
        jnp.dot(U,jnp.dot(jnp.diag(jnp.log(S)),V))
        
        log_val = jnp.matmul(jnp.matmul(P_nhalf, Q), P_nhalf)
        U,S,V = jnp.linalg.svd(log_val)
        log_val = jnp.dot(U,jnp.dot(jnp.diag(jnp.log(S)),V))
        
        w = jnp.matmul(jnp.matmul(P_phalf, log_val), P_phalf)
            
        return jnp.dot(self.invJF((x[1],x[1])),w.reshape(-1))
    
    def StdLogEmbedded(self, x:Tuple[Array, Array], y:Array)->Array:
        
        P = self.F(x).reshape(self.N,self.N)
        S = y.reshape(self.N,self.N)
        
        P_phalf = jnp.real(jscipy.linalg.sqrtm(P))
        P_nhalf = jnp.real(jscipy.linalg.sqrtm(jnp.linalg.inv(P)))
        
        w = jnp.matmul(jnp.matmul(P_phalf, \
                                     logm(jnp.matmul(jnp.matmul(P_nhalf, S), P_nhalf))),
                          P_phalf)
            
        return w
    
    def StdParallelTransport(self, x:Tuple[Array, Array], y:Tuple[Array, Array], v:Array)->Array:
        
        P1 = self.F(x).reshape(self.N,self.N)
        P2 = self.F(y).reshape(self.N,self.N)
        v = v.reshape(self.N,self.N)
        
        P_phalf = jnp.real(jscipy.linalg.sqrtm(P))
        P_nhalf = jnp.real(jscipy.linalg.sqrtm(jnp.linalg.inv(P)))
        
        logxy = self.StdLogEmbedded(x,P2)
        expxy = jscipy.linalg.expm(0.5*jnp.matmul(jnp.matmul(P1_nhalf, logxy), P1_nhalf))
        
        psi = jnp.matmul(jnp.matmul(P1_nhalf, v), P1_nhalf)
        term1 = jnp.matmul(P1_phalf, expxy)
        term2 = jnp.matmul(expxy, P1_phalf)
        
        return jnp.matmul(jnp.matmul(term1, psi), term2)
    
    def StdDist(self, x:Tuple[Array, Array], y:Tuple[Array,Array])->Array:
        
        P1 = self.F(x).reshape(self.N,self.N)
        P2 = self.F(y).reshape(self.N,self.N)
        
        U, S, Vh = jnp.linalg.svd(jnp.linalg.solve(P1, P2))
        
        return jnp.sqrt(jnp.sum(jnp.log(S)**2))

    def StdDot(self, x:Tuple[Array, Array], v:Array, w:Array)->Array:
        
        P = self.F(x).reshape(self.N,self.N)
        
        v1 = jnp.linalg.solve(P, v)
        v2 = jnp.linalg.solve(p, w)
        
        return jnp.trace(jnp.matmul(v1,v2))

    def StdProj(self, x:Array, v:Array) -> Array:
        
        P = x.reshape(self.N,self.N)
        v = v.reshape(self.N, self.N)
        
        v_symmetric = 0.5*(v+v.T)
        
        return v_symmetric.reshape(-1)#jnp.matmul(jnp.matmul(P1_phalf, v_symmetric), P1_phalf).reshape(-1)
    
    def DupMat(self, N:int):
        
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
            D, _, _, _ = lax.scan(lambda carry, i: step_col(carry, i, j), init=(D,A,u,j), xs=jnp.arange(0,N-j,1))[0]
        
        return D.T

    def invDupMat(self, N:int):
        
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
        
        D, _, _, _ = lax.scan(lambda carry, i: step_col(carry, i, 0, 1.0), init=(D,A,u,0), xs=jnp.arange(0,N,1))[0]
        for j in range(1,N):
            D, _, _, _ = lax.scan(lambda carry, i: step_col(carry, i, j, 0.5), init=(D,A,u,j), xs=jnp.arange(0,N-j,1))[0]
        
        return D
    
    def centered_chart(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return lax.stop_gradient(self.F(x))
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