#!/bin/sh
#BSUB -q gpuv100
#BSUB -J HypParaboloid2_st
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u fmry@dtu.dk
#BSUB -B
#BSUB -N

#Load the following in case
#module load python/3.8
module swap cuda/12.0
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module swap python3/3.10.12

python3 train_t.py \
    --manifold HypParaboloid \
    --dim 2 \
    --load_model 0 \
    --max_T 1.0 \
    --lr_rate 0.0002 \
    --epochs 50000 \
    --x_samples 64 \
    --repeats 16 \
    --samples_per_batch 32 \
    --dt_steps 100 \
    --save_step 100 \
    --seed 2712
