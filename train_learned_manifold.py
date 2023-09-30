#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:44:27 2023

@author: fmry
"""

#%% Modules

from ManLearn.train_MNIST import train_vae as train_mnist
from ManLearn.train_SVHN import train_vae as train_svhn
from ManLearn.train_CelebA import train_vae as train_celeba
#from ManLearn.train_CelebA import train_vae

#train_svhn()
#train_mnist()
train_celeba()