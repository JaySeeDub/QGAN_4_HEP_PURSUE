#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -e slurm_debug_%j.err

export SLURM_CPU_BIND="cores"
module load conda
conda activate myenv

srun python full_pipeline.py \
    --jet-images ../data/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5 \
    --datasets-path datasets.pt \
    --train-on signal \
    --sampler kde \
    --model-type classical \
    --sigma 0.1 \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.001 \
    --debug 1 \
    --save-model
