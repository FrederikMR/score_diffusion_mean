#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Euclidean20_s1dsm
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u fmry@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o sendmeemail/error_%J.out 
#BSUB -e sendmeemail/output_%J.err 

#Load the following in case
#module load python/3.8
module swap cuda/12.0
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module swap python3/3.10.12

python3 train_score.py \
    --manifold Euclidean \
    --dim 20 \
    --s1_loss_type dsm \
    --s2_loss_type dsm \
    --load_model 0 \
    --T_sample 0 \
    --t0 0.01 \
    --train_net s1 \
    --max_T 1.0 \
    --lr_rate 0.001 \
    --epochs 50000 \
    --warmup_epochs 1000 \
    --x_samples 128 \
    --t_samples 100 \
    --repeats 128 \
    --dt_steps 100 \
    --save_step 100 \
    --seed 2712
