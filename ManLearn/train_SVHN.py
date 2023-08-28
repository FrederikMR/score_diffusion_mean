#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:14:54 2023

@author: fmry
"""

#%% Sources

#https://github.com/deepmind/dm-haiku/blob/main/examples/vae.py

#%% Modules

#ManLearn
from ManLearn.initialize import *
from ManLearn.VAE.VAE_SVHN import model, VAEOutput
from ManLearn.model_loader import save_model

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--save_path', default="ManLearn/models/SVHN/VAE/",
                        type=str)
    parser.add_argument('--save_step', default=100,
                        type=int)

    #Hyper-parameters
    parser.add_argument('--split', default="train[:80%]", 
                        type=str)
    parser.add_argument('--batch_size', default=100, 
                        type=int)
    parser.add_argument('--seed', default=2712, 
                        type=int)
    parser.add_argument('--lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--epochs', default=50000,
                        type=int)

    args = parser.parse_args()
    
    return args

#%% Training batch

class Batch(NamedTuple):
    image: Array  # [B, H, W, C]x
  
#%% Training State

class TrainingState(NamedTuple):
    params: hk.Params
    state_val: dict
    opt_state: optax.OptState
    rng_key: Array
  
#%% Load Dataset

def load_dataset(split: str='train[:80%]', batch_size: int=100, seed: int=2712) -> Iterator[Batch]:
  ds = (
      tfds.load("svhn_cropped", split=split, data_dir="../Data/", download=False)
      .shuffle(buffer_size=10 * batch_size, seed=seed)
      .batch(batch_size)
      .prefetch(buffer_size=5)
      .repeat()
      .as_numpy_iterator()
  )
  return map(lambda x: Batch(x["image"]/255.0), ds)
    
#%% Main

def train_vae():
    
    args = parse_args()

    @jit
    def gaussian_likelihood(x, x_hat):
        
        return jnp.mean((x-x_hat)**2)
    
    @jit
    def kl_divergence(mu, logvar):
        
        return -0.5*jnp.mean(jnp.sum(1+logvar-mu**2-jnp.exp(logvar), axis=1), axis=0)
    
    @jit
    def loss_fn(params, state_val, rng_key, batch:Batch)->Array:
        
        out, state_val = model.apply(params, state_val, rng_key, batch.image)
        z, x_hat, mu, logvar = out
        
        rec_loss = gaussian_likelihood(batch.image, x_hat)
        kld = kl_divergence(mu, logvar)
        
        elbo = rec_loss+kld
        
        return elbo, (state_val, kld, rec_loss, elbo)
    
    @jit
    def update(state: TrainingState, batch: Batch) -> TrainingState:
      """Performs a single SGD step."""
      
      rng_key, next_rng_key = jran.split(state.rng_key)
      gradients, aux = grad(loss_fn, has_aux=True)(state.params, state.state_val, rng_key, batch)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      
      return TrainingState(new_params, aux[0], new_opt_state, next_rng_key), aux[1:]
    
    # Load datasets.
    train_dataset = load_dataset(args.split, args.batch_size, args.seed)
    optimizer = optax.adam(args.lr_rate)
    
    # Initialise the training state.
    initial_rng_key = jran.PRNGKey(args.seed)
    initial_params, state_val = model.init(initial_rng_key, next(train_dataset).image)
    initial_opt_state = optimizer.init(initial_params)
    state = TrainingState(initial_params, state_val, initial_opt_state, initial_rng_key)
    
    # Run training and evaluation.
    kld_loss = []
    rec_loss = []
    elbo_loss = []
    for step in range(args.epochs):
        ds = next(train_dataset)
        state, loss = update(state, ds)
    
        if step % args.save_step == 0:
               
            kld_loss.append(loss[0])
            rec_loss.append(loss[1])
            elbo_loss.append(loss[2])
            
            file_name = os.path.join(args.save_path, "rec_loss.npy")
            np.save(file_name, jnp.stack(rec_loss))
            
            file_name = os.path.join(args.save_path, "kld_loss.npy")
            np.save(file_name, jnp.stack(kld_loss))
            
            file_name = os.path.join(args.save_path, "elbo_loss.npy")
            np.save(file_name, jnp.stack(elbo_loss))
                        
            save_model(args.save_path, state)
            
            print("Epoch: {} \t KLD = {:.4f} \t Log Likelihood = {:.4f} \t ELBO = {:.4f} ".format(step,
                                                                                        loss[0],
                                                                                        loss[1],
                                                                                        loss[2]))
    kld_loss.append(loss[0])
    rec_loss.append(loss[1])
    elbo_loss.append(loss[2])
    
    file_name = os.path.join(args.save_path, "rec_loss.npy")
    np.save(file_name, jnp.stack(rec_loss))
    
    file_name = os.path.join(args.save_path, "kld_loss.npy")
    np.save(file_name, jnp.stack(kld_loss))
    
    file_name = os.path.join(args.save_path, "elbo_loss.npy")
    np.save(file_name, jnp.stack(elbo_loss))
                
    save_model(args.save_path, state)
    
    return


