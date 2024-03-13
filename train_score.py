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
from jax import Array

from scipy import ndimage

from gp.gp import RM_EG

#haiku
import haiku as hk

#argparse
import argparse

#scores
from models import models

from typing import NamedTuple

#os
import os

#jaxgeometry
from jaxgeometry.manifolds import *
from jaxgeometry.statistics.score_matching import train_s1, train_s2, train_s1s2, TMSampling, LocalSampling, \
    EmbeddedSampling, ProjectionSampling
from jaxgeometry.statistics.score_matching.model_loader import load_model
from ManLearn.train_MNIST import load_dataset as load_mnist

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="gp_mnist",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--loss_type', default="dsmvr",
                        type=str)
    parser.add_argument('--load_model', default=0,
                        type=int)
    parser.add_argument('--T_sample', default=0,
                        type=int)
    parser.add_argument('--t', default=0.1,
                        type=float)
    parser.add_argument('--gamma', default=1.0,
                        type=float)
    parser.add_argument('--train_net', default="s1",
                        type=str)
    parser.add_argument('--max_T', default=1.0,
                        type=float)
    parser.add_argument('--lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--epochs', default=50000,
                        type=int)
    parser.add_argument('--x_samples', default=32,
                        type=int)
    parser.add_argument('--t_samples', default=128,#128
                        type=int)
    parser.add_argument('--repeats', default=8,
                        type=int)
    parser.add_argument('--samples_per_batch', default=16,
                        type=int)
    parser.add_argument('--dt_steps', default=1000,
                        type=int)
    parser.add_argument('--save_step', default=1,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% train for (x,y,t)

def train_score()->None:
    
    args = parse_args()
    
    N_sim = args.x_samples*args.repeats
    T_sample_name = (args.T_sample == 1)*"T"
    if args.loss_type == "dsmdiagvr":
        s1_path = f"scores/{args.manifold}{args.dim}/s1{T_sample_name}_dsmvr/"
    elif args.loss_type == "dsmdiag":
        s1_path = f"scores/{args.manifold}{args.dim}/s1{T_sample_name}_dsm/"
    else:
        s1_path = f"scores/{args.manifold}{args.dim}/s1{T_sample_name}_{args.loss_type}/"
    s2_path = f"scores/{args.manifold}{args.dim}/s2{T_sample_name}_{args.loss_type}/"
    s1s2_path = f"scores/{args.manifold}{args.dim}/s1s2{T_sample_name}_{args.loss_type}/"
    
    if args.manifold == "Euclidean":
        sampling_method = 'LocalSampling'
        M = Euclidean(N=args.dim)
        generator_dim = M.dim
        x0 = M.coords([0.]*args.dim)
    elif args.manifold == "Circle":
        sampling_method = 'TMSampling'
        M = S1()
        generator_dim = M.emb_dim
        x0 = M.coords([0.])
    elif args.manifold == "Sphere":
        sampling_method = 'TMSampling'
        M = nSphere(N=args.dim)
        generator_dim = M.emb_dim
        x0 = M.coords([0.]*args.dim)
    elif args.manifold == "HyperbolicSpace":
        sampling_method = 'TMSampling'
        M = nHyperbolicSpace(N=args.dim)
        generator_dim = M.emb_dim
        x0 = (jnp.concatenate((jnp.zeros(args.dim-1), -1.*jnp.ones(1))),)*2
    elif args.manifold == "Grassmanian":
        sampling_method = 'TMSampling'
        M = Grassmanian(N=2*args.dim,K=args.dim)
        generator_dim = M.emb_dim
        x0 = (jnp.eye(2*args.dim)[:,:args.dim].reshape(-1),)*2
    elif args.manifold == "SO":
        sampling_method = 'TMSampling'
        M = SO(N=args.dim)
        generator_dim = M.emb_dim
        x0 = (jnp.eye(args.dim).reshape(-1),)*2
    elif args.manifold == "Stiefel":
        sampling_method = 'TMSampling'
        M = Stiefel(N=args.dim, K=2)
        generator_dim = M.emb_dim
        x0 = (jnp.block([jnp.eye(2), jnp.zeros((2,args.dim-2))]).T.reshape(-1),)*2
    elif args.manifold == "Ellipsoid":
        sampling_method = 'TMSampling'
        M = nEllipsoid(N=args.dim, params = jnp.linspace(0.5,1.0,args.dim+1))
        generator_dim = M.emb_dim
        x0 = M.coords([0.]*args.dim)
    elif args.manifold == "Cylinder":
        sampling_method = 'EmbeddedSampling'
        M = Cylinder(params=(1.,jnp.array([0.,0.,1.]),jnp.pi/2.))
        generator_dim = M.dim
        x0 = M.coords([0.]*2)
    elif args.manifold == "Torus":
        sampling_method = 'EmbeddedSampling'
        M = Torus()
        generator_dim = M.dim
        x0 = M.coords([0.]*2)
    elif args.manifold == "Landmarks":
        sampling_method = 'LocalSampling'
        M = Landmarks(N=args.dim,m=2)   
        generator_dim = M.dim
        x0 = M.coords(jnp.vstack((jnp.linspace(-5.0,0.0,M.N),jnp.linspace(5.0,-0.0,M.N))).T.flatten())
        if args.dim >=10:
            with open('../../Data/landmarks/Papilonidae/Papilionidae_landmarks.txt', 'r') as the_file:
                all_data = [line.strip() for line in the_file.readlines()]
                
                x1 = jnp.array([float(x) for x in all_data[0].split()[2:]])
                x2 = jnp.array([float(x) for x in all_data[1].split()[2:]])
                
                idx = jnp.round(jnp.linspace(0, len(x1) - 1, args.dim)).astype(int)
                x0 = M.coords(jnp.vstack((x1[idx],x2[idx])).T.flatten())
    elif args.manifold == "SPDN":
        sampling_method = 'LocalSampling'
        M = SPDN(N=args.dim)
        generator_dim = M.dim
        x0 = M.coords([10.]*(args.dim*(args.dim+1)//2))
    elif args.manifold == "Sym":
        sampling_method = 'LocalSampling'
        M = Sym(N=args.dim)
        generator_dim = M.dim
        x0 = M.coords([1.]*(args.dim*(args.dim+1)//2))
    elif args.manifold == "HypParaboloid":
        sampling_method = 'LocalSampling'
        M = HypParaboloid()
        generator_dim = M.dim
        x0 = M.coords([0.]*2)
    elif args.manifold == 'gp_mnist':
        
        default_omega = 500.
        
        def k_fun(x,y, beta=1.0, omega=default_omega):
    
            x_diff = x-y
            
            return beta*jnp.exp(-omega*jnp.dot(x_diff, x_diff)/2)

        def Dk_fun(x,y, beta=1.0, omega=default_omega):
            
            x_diff = y-x
            
            return omega*x_diff*k_fun(x,y,beta,omega)
        
        def DDk_fun(x,y, beta=1.0, omega=default_omega):
            
            N = len(x)
            x_diff = (x-y).reshape(1,-1)
            
            return -omega*k_fun(x,y,beta,omega)*(x_diff.T.dot(x_diff)*omega-jnp.eye(N))
        
        rot = jnp.load('Data/MNIST/rot.npy')
        num_rotate = len(rot)

        theta = jnp.linspace(0,2*jnp.pi,num_rotate)
        x1 = jnp.cos(theta)
        x2 = jnp.sin(theta)
        
        sigman = 0.0
        X_training = jnp.vstack((x1,x2))
        y_training = rot.reshape(num_rotate, -1).T
        RMEG = RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, 
                     Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)

        g = lambda x: RMEG.G(x[0])
        
        M = LearnedManifold(g,N=2)
        generator_dim = M.dim
        x0 = M.coords(jnp.array([jnp.cos(0.), jnp.sin(0.)]))
        sampling_method = 'LocalSampling'
    else:
        return
        
    if sampling_method == 'LocalSampling':
        generator_dim = M.dim
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
        generator_dim = M.dim
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
        generator_dim = M.emb_dim
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
        generator_dim = M.emb_dim
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
        
    if generator_dim<10:
        layers = [50,100,100,50]
    elif generator_dim<50:
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
            models.MLP_s2(layers_alpha=layers, 
                          layers_beta=layers,
                          dim=generator_dim,
                          r = max(generator_dim//2,1))
            )
        
        return s1s2(x)

    if args.train_net == "s2":
        state = load_model(s1_path)
        rng_key = jran.PRNGKey(2712)
        s1 = lambda x,y,t: s1_model.apply(state.params,rng_key, jnp.hstack((x, y, t)))
        
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
                 dW_dim=generator_dim,
                 batch_size=batch_size,
                 state=state_s2,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 save_step=args.save_step,
                 save_path=s2_path,
                 seed=args.seed,
                 loss_type = args.loss_type
                 )
    elif args.train_net == "s1s2":
        state = load_model(s1_path)
        rng_key = jran.PRNGKey(2712)
            
        if not os.path.exists(s1s2_path):
            os.makedirs(s1s2_path)#s1s2 = hk.transform(lambda x: models.MLP_s1s2(models.MLP_s1(dim=M.dim, layers=layers), 
#                                              models.MLP_s2(layers_alpha=layers, layers_beta=layers,
#                                                dim=M.dim, r = max(M.dim//2,1)))(x))
            
        if args.load_model:
            state_s1s2 = load_model(s1s2_path)
        else:
            state_s1s2 = None
        
        train_s1s2(M=M,
                 s1s2_model=s1s2_model,
                 generator=data_generator,
                 N_dim=generator_dim,
                 dW_dim=generator_dim,
                 batch_size=batch_size,
                 state=state_s1s2,
                 s1_params=state.params,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 save_step=args.save_step,
                 gamma=args.gamma,
                 save_path=s1s2_path,
                 seed=args.seed,
                 loss_type = args.loss_type
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
                 dW_dim=generator_dim,
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
    
