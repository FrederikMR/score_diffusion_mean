#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:11:07 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax
import jax.numpy as jnp
import jax.random as jran

#haiku
import haiku as hk

#argparse
import argparse

#scores
from models import models

#os
import os

#jaxgeometry
from jaxgeometry.manifolds import Euclidean, nSphere, nEllipsoid, Cylinder, S1, Torus, \
    H2, Landmarks, Heisenberg, SPDN, Latent, HypParaboloid, Sym, nHyperbolicSpace, Grassmanian
from jaxgeometry.statistics.score_matching import train_s1, train_s2, train_s1s2, TMSampling, LocalSampling, \
    EmbeddedSampling, ProjectionSampling
from jaxgeometry.statistics.score_matching.model_loader import load_model

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="SN",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--loss_type', default="dsm",
                        type=str)
    parser.add_argument('--load_model', default=0,
                        type=int)
    parser.add_argument('--T_sample', default=0,
                        type=int)
    parser.add_argument('--t', default=0.1,
                        type=float)
    parser.add_argument('--train_net', default="s2",
                        type=str)
    parser.add_argument('--max_T', default=1.0,
                        type=float)
    parser.add_argument('--lr_rate', default=0.001,
                        type=float)
    parser.add_argument('--epochs', default=50000,
                        type=int)
    parser.add_argument('--x_samples', default=32*2,
                        type=int)
    parser.add_argument('--t_samples', default=128*2,
                        type=int)
    parser.add_argument('--repeats', default=8*2,
                        type=int)
    parser.add_argument('--samples_per_batch', default=16*2,
                        type=int)
    parser.add_argument('--dt_steps', default=1000,
                        type=int)
    parser.add_argument('--save_step', default=10,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% train for (x,y,t)

def train_score()->None:
    
    args = parse_args()
    
    N_sim = args.x_samples*args.repeats
    
    if args.manifold == "RN":
        sampling_method = 'LocalSampling'
        generator_dim = args.dim
        if not args.T_sample:
            s1_path = ''.join(('scores/R',str(args.dim),'/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/R',str(args.dim),'/s2/'))
        else:
            s1_path = ''.join(('scores/R',str(args.dim),'/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/R',str(args.dim),'/s2_T'))
        s1s2_path = ''.join(('scores/R',str(args.dim),'/s1s2/'))
            
        M = Euclidean(N=args.dim)
        x0 = M.coords([0.]*args.dim)
        
        if args.dim<10:
            layers = [50,100,100,50]
        elif args.dim<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))

        @hk.transform
        def s1s2_model(x):
            
            s1s2 =  models.MLP_s1s2(
                models.MLP_s1(dim=generator_dim, layers=layers), 
                models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                                dim=generator_dim, 
                                                                r = max(generator_dim//2,1)))
            
            return s1s2(x)
    
    elif args.manifold == "Circle":
        sampling_method = 'LocalSampling'
        generator_dim=1
        if not args.T_sample:
            s1_path = ''.join(('scores/S1/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/S1/s2/'))
        else:
            s1_path = ''.join(('scores/S1/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/S1/s2_T'))
        
        M = S1()
        x0 = M.coords([0.])
        
        layers = [50,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "SN":
        sampling_method = 'TMSampling'
        generator_dim = args.dim+1
        if not args.T_sample:
            s1_path = ''.join(('scores/S',str(args.dim),'/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/S',str(args.dim),'/s2/'))
        else:
            s1_path = ''.join(('scores/S',str(args.dim),'/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/S',str(args.dim),'/s2_T'))
            
        M = nSphere(N=args.dim)
        x0 = M.coords([0.]*args.dim)
        
        if args.dim<10:
            layers = [50,100,100,50]
        elif args.dim<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "HyperbolicSpace":
        sampling_method = 'TMSampling'
        generator_dim = args.dim
        if not args.T_sample:
            s1_path = ''.join(('scores/HyperbolicSpace',str(args.dim),'/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/HyperbolicSpace',str(args.dim),'/s2/'))
        else:
            s1_path = ''.join(('scores/HyperbolicSpace',str(args.dim),'/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/HyperbolicSpace',str(args.dim),'/s2_T'))
            
        M = nHyperbolicSpace(N=args.dim)
        x0 = (jnp.concatenate((jnp.zeros(args.dim-1), -1.*jnp.ones(1))),)*2
        
        if args.dim<10:
            layers = [50,100,100,50]
        elif args.dim<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "Grassmanian":
        sampling_method = 'TMSampling'
        generator_dim = 2*args.dim*args.dim
        if not args.T_sample:
            s1_path = ''.join(('scores/Grassmanian',str(args.dim),'/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Grassmanian',str(args.dim),'/s2/'))
        else:
            s1_path = ''.join(('scores/Grassmanian',str(args.dim),'/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Grassmanian',str(args.dim),'/s2_T'))
            
        M = Grassmanian(N=2*args.dim,K=args.dim)
        x0 = (jnp.eye(2*args.dim)[:,:args.dim].reshape(-1),)*2
        
        if args.dim<10:
            layers = [50,100,100,50]
        elif args.dim<50:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "Ellipsoid":
        sampling_method = 'TMSampling'
        generator_dim = args.dim+1
        if not args.T_sample:
            s1_path = ''.join(('scores/Ellipsoid',str(args.dim),'/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Ellipsoid',str(args.dim),'/s2/'))
        else:
            s1_path = ''.join(('scores/Ellipsoid',str(args.dim),'/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Ellipsoid',str(args.dim),'/s2_T'))

        M = nEllipsoid(N=args.dim, params = jnp.linspace(0.5,1.0,args.dim+1))
        x0 = M.coords([0.]*args.dim)
        
        if args.dim<10:
            layers = [50,100,200,200,100,50]
        elif args.dim<50:
            layers = [50,100,200,400,400,200,100,50]
        else:
            layers = [50,100,200,400,800,800,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "Cylinder":
        sampling_method = 'EmbeddedSampling'
        generator_dim = 3
        if not args.T_sample:
            s1_path = ''.join(('scores/Cylinder/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Cylinder/s2/'))
        else:
            s1_path = ''.join(('scores/Cylinder/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Cylinder/s2_T'))
        

        M = Cylinder(params=(1.,jnp.array([0.,0.,1.]),jnp.pi/2.))
        x0 = M.coords([0.]*2)
        
        layers = [50,100,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "Torus":
        sampling_method = 'EmbeddedSampling'
        generator_dim = 3
        if not args.T_sample:
            s1_path = ''.join(('scores/Torus/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Torus/s2/'))
        else:
            s1_path = ''.join(('scores/Torus/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Torus/s2_T'))

        M = Torus()        
        x0 = M.coords([0.]*2)
        
        layers = [50,100,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "Landmarks":
        sampling_method = 'LocalSampling'
        generator_dim = 2*args.dim
        if not args.T_sample:
            s1_path = ''.join(('scores/Landmarks',str(args.dim),'/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Landmarks',str(args.dim),'/s2/'))
        else:
            s1_path = ''.join(('scores/Landmarks',str(args.dim),'/s1_T_/',args.loss_type,'/'))
            s2_path = ''.join(('scores/Landmarks',str(args.dim),'/s2_T'))

        M = Landmarks(N=args.dim,m=2)
        
        x0 = M.coords(jnp.vstack((jnp.linspace(-10.0,10.0,M.N),jnp.linspace(10.0,-10.0,M.N))).T.flatten())
        #x0 = M.coords(jnp.vstack((jnp.linspace(-5.0,5.0,M.N),jnp.zeros(M.N))).T.flatten())
        
        if args.dim >=10:
            with open('../../Data/landmarks/Papilonidae/Papilionidae_landmarks.txt', 'r') as the_file:
                all_data = [line.strip() for line in the_file.readlines()]
                
                x1 = jnp.array([float(x) for x in all_data[0].split()[2:]])
                x2 = jnp.array([float(x) for x in all_data[1].split()[2:]])
                
                #idx = jnp.round(jnp.linspace(0, len(x1) - 1, args.dim)).astype(int)
                x0 = M.coords(jnp.vstack((x1[::len(x1)//args.dim],x2[::len(x2)//args.dim])).T.flatten())
        
        if args.dim<5:
            layers = [50,100,100,50]
            #layers = [50,100,200,200,100,50]
        elif args.dim<25:
            layers = [50,100,200,400,400,200,100,50]
        else:
            layers = [50,100,200,400,800,800,400,200,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "SPDN":
        sampling_method = 'LocalSampling'
        generator_dim = (args.dim*(args.dim+1))//2
        if not args.T_sample:
            s1_path = ''.join(('scores/SPDN',str(args.dim),'/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/SPDN',str(args.dim),'/s2/'))
        else:
            s1_path = ''.join(('scores/SPDN',str(args.dim),'/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/SPDN',str(args.dim),'/s2_T'))

        M = SPDN(N=args.dim)
        x0 = M.coords([10.]*(args.dim*(args.dim+1)//2))
        
        if args.dim<3:
            layers = [50,100,100,50]
        elif args.dim<8:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]

        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "Sym":
        sampling_method = 'LocalSampling'
        generator_dim = (args.dim*(args.dim+1))//2
        if not args.T_sample:
            s1_path = ''.join(('scores/Sym',str(args.dim),'/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/Sym',str(args.dim),'/s2/'))
        else:
            s1_path = ''.join(('scores/Sym',str(args.dim),'/s1_T_/',args.loss_type,'/'))
            s2_path = ''.join(('scores/Sym',str(args.dim),'/s2_T'))

        M = Sym(N=args.dim)
        x0 = M.coords([10.]*(args.dim*(args.dim+1)//2))
        
        if args.dim<3:
            layers = [50,100,100,50]
        elif args.dim<8:
            layers = [50,100,200,200,100,50]
        else:
            layers = [50,100,200,400,400,200,100,50]

        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
        
    elif args.manifold == "HypParaboloid":
        sampling_method = 'LocalSampling'
        generator_dim = 2
        if not args.T_sample:
            s1_path = ''.join(('scores/HypParaboloid/s1_',args.loss_type,'/'))
            s2_path = ''.join(('scores/HypParaboloid/s2/'))
        else:
            s1_path = ''.join(('scores/HypParaboloid/s1_T_',args.loss_type,'/'))
            s2_path = ''.join(('scores/HypParaboloid/s2_T'))

        M = HypParaboloid()
        x0 = M.coords([0.]*2)
        
        layers = [50,100,100,50]
        
        s1_model = hk.transform(lambda x: models.MLP_s1(dim=generator_dim, layers=layers)(x))
        s2_model = hk.transform(lambda x: models.MLP_s2(layers_alpha=layers, layers_beta=layers,
                                                        dim=generator_dim, 
                                                        r = max(generator_dim//2,1))(x))
    else:
        return
        
    if sampling_method == 'LocalSampling':
        dW_dim = M.dim
        data_generator = LocalSampling(M=M,
                                       x0=x0,
                                       repeats=args.repeats,
                                       x_samples=args.x_samples,
                                       t_samples=args.t_samples,
                                       N_sim=N_sim,
                                       max_T=args.max_T,
                                       dt_steps=args.dt_steps,
                                       T_sample=args.T_sample,
                                       t=args.t
                                       )
    elif sampling_method == "EmbeddedSampling":
        dW_dim = M.dim
        data_generator = EmbeddedSampling(M=M,
                                          x0=x0,
                                          repeats=args.repeats,
                                          x_samples=args.x_samples,
                                          t_samples=args.t_samples,
                                          N_sim=N_sim,
                                          max_T=args.max_T,
                                          dt_steps=args.dt_steps,
                                          T_sample=args.T_sample,
                                          t=args.t
                                          )
    elif sampling_method == "ProjectionSampling":
        dW_dim = M.emb_dim
        data_generator = ProjectionSampling(M=M,
                                            x0=(x0[1],x0[0]),
                                            dim=generator_dim,
                                            repeats=args.repeats,
                                            x_samples=args.x_samples,
                                            t_samples=args.t_samples,
                                            N_sim=N_sim,
                                            max_T=args.max_T,
                                            dt_steps=args.dt_steps,
                                            T_sample=args.T_sample,
                                            t=args.t
                                            )
    elif sampling_method == "TMSampling":
        dW_dim = M.emb_dim
        data_generator = TMSampling(M=M,
                                    x0=(x0[1],x0[0]),
                                    dim=generator_dim,
                                    Exp_map=lambda x, v: M.ExpEmbedded(x[0],v),
                                    repeats=args.repeats,
                                    x_samples=args.x_samples,
                                    t_samples=args.t_samples,
                                    N_sim=N_sim,
                                    max_T=args.max_T,
                                    dt_steps=args.dt_steps,
                                    T_sample=args.T_sample,
                                    t=args.t
                                    )
    
    if not os.path.exists('scores/output/'):
        os.makedirs('scores/output/')
        
    if not os.path.exists('scores/error/'):
        os.makedirs('scores/error/')
    
    if args.T_sample:
        batch_size = args.x_samples*args.repeats
    else:
        batch_size = args.x_samples*args.t_samples*args.repeats
    if args.train_net == "s2":
        state = load_model(s1_path)
        rng_key = jran.PRNGKey(2712)
        s1 = lambda x,y,t: s1_model.apply(state.params,rng_key, jnp.hstack((x, y, t)))
        #s1 = lambda x,y,t: M.grady_log_hk(x,y,t)
        
        if args.load_model:
            state_s2 = load_model(s1_path)
        else:
            state_s2 = None            

        if not os.path.exists(s2_path):
            os.makedirs(s2_path)
        
        train_s2(M=M,
                 s1_model=s1,
                 s2_model=s2_model,
                 generator=data_generator,
                 N_dim=generator_dim,
                 dW_dim=dW_dim,
                 batch_size=batch_size,
                 state=state_s2,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 save_step=args.save_step,
                 save_path=s2_path,
                 seed=args.seed
                 )
    elif args.train_net == "s1s2":
        state = load_model(s1_path)
        rng_key = jran.PRNGKey(2712)
            
        if not os.path.exists(s1s2_path):
            os.makedirs(s1s2_path)
            
        if args.load_model:
            state_s1s2 = load_model(s1s2_path)
        else:
            state_s1s2 = None
        
        train_s1s2(M=M,
                 s1s2_model=s1s2_model,
                 generator=data_generator,
                 N_dim=generator_dim,
                 dW_dim=dW_dim,
                 batch_size=batch_size,
                 state=state_s1s2,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 save_step=args.save_step,
                 save_path=s1s2_path,
                 seed=args.seed
                 )
    else:
        if args.load_model:
            state = load_model(s1_path)
        else:
            state = None

        if not os.path.exists(s1_path):
            os.makedirs(s1_path)
            
        train_s1(M=M,
                 model=s1_model,
                 generator=data_generator,
                 N_dim=generator_dim,
                 dW_dim=dW_dim,
                 batch_size=batch_size,
                 state =state,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 save_step=args.save_step,
                 save_path=s1_path,
                 loss_type=args.loss_type,
                 seed=args.seed
                 )
    
    
    return

#%% Main

if __name__ == '__main__':
        
    train_score()
    
