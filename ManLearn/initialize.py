#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:18:59 2023

@author: fmry
"""

#%% Modules

#jax
from jax import Array, jit, grad
from jax.tree_util import tree_leaves, tree_flatten, tree_unflatten, tree_map
import jax.numpy as jnp
import jax.random as jran

from jax.nn import elu, sigmoid, swish

#numpy 
import numpy as np

#pickle
import pickle

#haiku
import haiku as hk

#dataclasses
import dataclasses

#typing
from typing import Tuple, NamedTuple, Iterator

#optax
import optax

#tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

#os
import os

#parser
import argparse