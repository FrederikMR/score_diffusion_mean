#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 00:45:24 2023

@author: fmry
"""

#%% Sources

#%% Modules

#argparse
import argparse

#ManLearn
from ManLearn.train_MNIST import train_vae as train_mnist
from ManLearn.train_SVHN import train_vae as train_svhn
from ManLearn.train_CelebA import train_vae as train_celeba

#%% Args Parse

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="MNIST",
                        type=str)

    args = parser.parse_args()
    return args

#%% Code

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.manifold == "MNIST":
        train_mnist()
    elif args.manifold == "SVHN":
        train_svhn()
    elif args.manifold == "CelebA":
        train_celeba()
