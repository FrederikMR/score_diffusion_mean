#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:18:22 2023

@author: fmry
"""

#%% Import generally used model for JaxMan

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#numpy
import numpy as np

#JAX
import jax.numpy as jnp
import jax
from jax import grad, jacfwd, jit, vmap, Array, lax, random

#JAX Optimizer
from jax.example_libraries import optimizers

#minmize
from scipy.optimize import minimize,fmin_bfgs,fmin_cg

#typing
from typing import Callable, NamedTuple

#Plotting
import matplotlib.pyplot as plt

#scipy
import scipy

#sklearn
from sklearn.decomposition import PCA

#os
import os

from functools import partial

from JAXGeometry.utils import *

import itertools

import time


#pickle
import pickle