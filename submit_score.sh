#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Sphere3_s2vr
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u fmry@dtu.dk
#BSUB -o scores/output/hpc/output_%J.out
#BSUB -e scores/error/hpc/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
#module load python/3.8
module swap cuda/12.0
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module swap python3/3.10.12

python3 train_score.py \
    --manifold Sphere \
    --dim 3 \
    --train_net s1 \
    --s1_loss_type dsmvr \
    --s2_loss_type dsmvr \
    --epochs 50000 \
    --lr_rate 0.0002 \
    --T 1.0 \
    --dt_steps 1000 \
    --x_samples 64 \
    --t_samples 256 \
    --repeats 16 \
    --t0_sample 0 \
    --t0 0.01 \
    --gamma 1.0 \
    --load_model 0 \
    --save_step 100 \
    --seed 2712 \