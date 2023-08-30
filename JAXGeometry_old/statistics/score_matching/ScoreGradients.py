#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:15:38 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JAXGeometry.setup import *
from JAXGeometry.params import *

#%% Initialize

def initialize(M:object, 
               file_path:str=None):
    
    def load_model_fn():
        
        file_name = os.path.join(ckpt_dir, "epoch_{}_tree.pkl".format(epoch))
        with open(file_name, "rb") as f:
            tree_struct = pickle.load(f)
            
        leaves, treedef = tree_flatten(tree_struct)
        file_name = os.path.join(ckpt_dir, "epoch_{}_arrays.npy".format(epoch))
        with open(file_name, "rb") as f:
            flat_state = [np.load(f) for _ in leaves]

        return tree_unflatten(treedef, flat_state)
    
    return
