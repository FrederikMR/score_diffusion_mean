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

#Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#JAX
from jax import vmap, grad, jacfwd, jacrev, jit, value_and_grad, Array, device_get, \
    tree_leaves, tree_map, tree_flatten, tree_unflatten, lax, Array
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.optimize as jopt
import jax.random as jrandom

#from jax.config import config
#config.update('jax_enable_x64', False)

#JAX Optimization
from jax.example_libraries import optimizers

#Scipy
from scipy.linalg import logm

#random
import random

#haiku
import haiku as hk

#optax
import optax

#tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

#numpy
import numpy as np

#scipy
from scipy.optimize import minimize,fmin_bfgs,fmin_cg, approx_fprime

#sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

#os
import os

#pickle
import pickle

#dataclasses
import dataclasses

#Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.figure

#functools
from functools import partial

#typing
from typing import Callable, NamedTuple, Tuple, List

#%% Parameters

# default integration times and time steps
T = 1.
n_steps = 100
seed = 2712 #Old value 42

# Integrator variables:
default_method = 'euler'
#default_method = 'rk4'

global key
key = jrandom.PRNGKey(seed)

plt.rcParams['figure.figsize'] = 15,12


