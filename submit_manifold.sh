#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J SVHN
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u fmry@dtu.dk
#BSUB -o ManLearn/models/output/output_%J.out
#BSUB -e ManLearn/models/error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
module swap python3/3.9.11
module swap cuda/8.0
module swap cudnn/v7.0-prod-cuda8

python3 train_manifold.py \
    --manifold SVHN
