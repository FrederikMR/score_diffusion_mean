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
from ManLearn.VAE.VAE_CelebA import model, VAEOutput
from ManLearn.model_loader import save_model

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--save_path', default="ManLearn/models/CelebA/VAE/",
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
    opt_state: optax.OptState
    rng_key: Array
  
#%% Load Dataset

def load_dataset(data_dir:str = '../Data/CelebA/celeb_a/img_align_celeba',
                 img_size:Tuple[int, int] = (64, 64), batch_size: int=100, seed: int=2712) -> Iterator[Batch]:
    
    def preprocess_image(filename:str):
        
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, img_size)
        
        return image

    filenames = tf.constant([os.path.join(data_dir, fname) for fname in os.listdir(data_dir)])
    dataset = tf.data.Dataset.from_tensor_slices((filenames[:int(len(filenames)*0.8)]))

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.repeat().as_numpy_iterator()
    
    return map(lambda x: Batch(x), dataset)
    
#%% Main

def train_vae():
    
    args = parse_args()

    @jit
    def gaussian_likelihood(x, x_hat):
        
        return jnp.mean(jnp.square(x-x_hat))
    
    @jit
    def kl_divergence(mu, std):
        
        return -0.5*jnp.mean(jnp.sum(1+2.0*std-mu**2-jnp.exp(2.0*std), axis=-1))
    
    @jit
    def loss_fn(params, rng_key, batch:Batch)->Array:
        
        z, x_hat, mu, std = model.apply(params, rng_key, batch.image)
        
        rec_loss = gaussian_likelihood(batch.image, x_hat)
        kld = kl_divergence(mu, std)
        
        elbo = rec_loss+kld
        
        return elbo, (kld, rec_loss, elbo)
    
    @jit
    def update(state: TrainingState, batch: Batch) -> TrainingState:
      """Performs a single SGD step."""
      
      rng_key, next_rng_key = jran.split(state.rng_key)
      gradients, aux = grad(loss_fn, has_aux=True)(state.params, rng_key, batch)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      
      return TrainingState(new_params, new_opt_state, next_rng_key), aux
    
    # Load datasets.
    train_dataset = load_dataset()
    optimizer = optax.adam(args.lr_rate)
    
    # Initialise the training state.
    initial_rng_key = jran.PRNGKey(args.seed)
    initial_params = model.init(initial_rng_key, next(train_dataset).image)
    initial_opt_state = optimizer.init(initial_params)
    state = TrainingState(initial_params, initial_opt_state, initial_rng_key)

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


