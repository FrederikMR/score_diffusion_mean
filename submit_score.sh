#!/bin/sh
#BSUB -q gpua100
#BSUB -J Landmarks100_s1
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
    --manifold Landmarks \
    --dim 100 \
    --loss_type dsm \
    --load_model 0 \
    --T_sample 0 \
    --t 0.1 \
    --train_net s1 \
    --max_T 1.0 \
    --lr_rate 0.001 \
    --epochs 50000 \
    --x_samples 32 \
    --t_samples 128 \
    --repeats 8 \
    --samples_per_batch 16 \
    --dt_steps 1000 \
    --save_step 10 \
    --seed 2712
