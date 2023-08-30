#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:54:53 2023

@author: fmry
"""

#%% Modules

from JAXGeometry.setup import *

#%% Save model

def save_model(ckpt_dir: str, state) -> None:
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
      for x in jax.tree_leaves(state):
        np.save(f, x, allow_pickle=False)
    
    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
      pickle.dump(tree_struct, f)

#%% Load Model

def load_model(ckpt_dir:str):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
      tree_struct = pickle.load(f)
    
    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
      flat_state = [np.load(f) for _ in leaves]
    
    return jax.tree_unflatten(treedef, flat_state)