#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -e slurm_debug_%j.err

export SLURM_CPU_BIND="cores"
module load conda
conda activate myenv

srun python -u ./test_smoteGAN.py