#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Sphere5_s2
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u fmry@dtu.dk
#BSUB -o scores/output/output_%J.out
#BSUB -e scores/error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
#module load python/3.8
module swap cuda/12.0
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module swap python3/3.10.12

python3 train_score.py \
    --manifold Sphere \
    --dim 5 \
    --loss_type dsmdiag \
    --load_model 0 \
    --T_sample 0 \
    --t 0.01 \
    --train_net s2 \
    --max_T 1.0 \
    --lr_rate 0.0002 \
    --epochs 50000 \
    --x_samples 64 \
    --t_samples 256 \
    --repeats 16 \
    --samples_per_batch 32 \
    --dt_steps 1000 \
    --save_step 100 \
    --seed 2712
