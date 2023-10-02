#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:18:59 2023

@author: fmry
"""

#%% Modules

from jax import Array
import jax.numpy as jnp
from jax.nn import elu, sigmoid
import jax.random as jran

#haiku
import haiku as hk

#dataclasses
import dataclasses

#typing
from typing import Tuple, NamedTuple, Tuple