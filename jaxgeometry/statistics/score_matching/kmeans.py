#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:38:42 2023

@author: fmry
"""

#%% Sources

#https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670

#%% Modules

from jaxgeometry.setup import *

#%% Code

#def euclidean(point, data):
#    """
#    Euclidean distance between point & data.
#    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
#    """
#    return jnp.sqrt(jnp.sum((point - data)**2, axis=1))
class KMeans(object):
    def __init__(self, dist_fun, frechet_fun, n_clusters=4, max_iter=100):
        self.dist_fun = dist_fun
        self.frechet_fun = frechet_fun
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.key = jrandom.PRNGKey(2712)
    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        key, subkey = jrandom.split(self.key)
        self.key = subkey
        centroid_idx = [jrandom.choice(subkey, jnp.arange(0,len(X_train[0]), 1))]
        self.centroids = (X_train[0][jnp.array(centroid_idx)].reshape(1,-1), 
                          X_train[1][jnp.array(centroid_idx)].reshape(1,-1))
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = vmap(lambda x,chartx: vmap(lambda y,charty: \
                                               self.dist_fun((x,chartx), 
                                                             (y,charty)))(X_train[0], 
                                                                          X_train[1]))(self.centroids[0], 
                                                                                       self.centroids[1])
            dists = jnp.nan_to_num(jnp.sum(dists, axis=0), nan=0)         
            # Normalize the distances
            dists /= jnp.sum(dists)
            # Choose remaining points based on their distances
            key, subkey = jrandom.split(self.key)
            self.key = subkey
            new_centroid_idx = [jrandom.choice(key, jnp.arange(0,len(X_train[0])), p=dists)]
            centroid_idx += new_centroid_idx
            self.centroids = (X_train[0][jnp.array(centroid_idx)], 
                              X_train[1][jnp.array(centroid_idx)])
            
        for _ in range(self.max_iter):
            # Sort each datapoint, assigning to nearest centroid
            print(f"Iteration {_+1}/{self.max_iter}")
            sorted_points = [[] for _ in range(self.n_clusters)]
            dists = vmap(lambda x,chartx: vmap(lambda y,charty: \
                                               self.dist_fun((x,chartx), 
                                                             (y,charty)))(X_train[0], 
                                                                          X_train[1]))(self.centroids[0], 
                                                                                       self.centroids[1])
            centroid_idx = jnp.argmin(dists, axis=0)
            centroid_idx = jnp.stack([centroid_idx==k for k in range(self.n_clusters)])
            prev_centroids = self.centroids
            mu1 = []
            mu2 = []
            for i in range(self.n_clusters):
                mu = self.frechet_fun((X_train[0][centroid_idx[i]],
                                                    X_train[1][centroid_idx[i]]),
                                                   (self.centroids[0][i], self.centroids[1][i]))
                mu1.append(mu[0])
                mu2.append(mu[1])
            self.centroids = (jnp.stack(mu1), jnp.stack(mu2))
                
                
            for i in range(self.n_clusters):
                if jnp.isnan(self.centroids[0][i]).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids = list(self.centroids)
                    self.centroids[0] = self.centroids[0].at[i].set(prev_centroids[0][i])
                    self.centroids[1] = self.centroids[1].at[i].set(prev_centroids[1][i])
                    self.centroids = tuple(self.centroids)
                    
        self.centroid_idx = centroid_idx