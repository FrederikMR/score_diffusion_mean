#!/bin/sh
#BSUB -q compute
#BSUB -J SPDN10T
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u fmry@dtu.dk
#BSUB -o scores/output/output_%J.out
#BSUB -e scores/error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
#module load python/3.8
module swap python 3/3.9.11
module swap cuda/8.0
module swap cudnn/v7.0-prod-cuda8

python3 train_score.py \
    --manifold SPDN \
    --N 10 \
    --loss_type T \
    --train_net s1 \
    --max_T 0.1 \
    --lr_rate 0.001 \
    --epochs 50000 \
    --x_samples 32 \
    --t_samples 128 \
    --repeats 8 \
    --samples_per_batch 16 \
    --dt_steps 1000 \
    --save_step 10 \
    --seed 2712
