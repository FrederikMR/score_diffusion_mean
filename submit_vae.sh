#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Circle3D_vae
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u fmry@dtu.dk
#BSUB -o vae/output/output_%J.out
#BSUB -e vae/error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
#module load python/3.8
module swap cuda/12.0
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module swap python3/3.10.12

python3 train_vae.py \
    --data Circle3D \
    --data_path data/vae/ \
    --score_loss_type dsmvr \
    --training_type score \
    --sample_method Local \
    --max_T 1.0 \
    --vae_lr_rate 0.0002 \
    --score_lr_rate 0.0002 \
    --latent_dim 2 \
    --epochs 50000 \
    --vae_batch 100 \
    --vae_epochs 300 \
    --score_epochs 100 \
    --vae_split 0.0 \
    --score_x_samples 32 \
    --score_t_samples 128 \
    --score_repeats 8 \
    --dt_steps 1000 \
    --save_step 100 \
    --save_path vae/joint_train/ \
    --vae_save_path vae/pretrain_vae/ \
    --score_save_path score/pretrain_vae/ \
    --seed 2712
