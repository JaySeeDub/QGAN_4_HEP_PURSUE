#!/bin/bash
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=$SCRATCH/QuGAN/logs/slurm-%j.out
#SBATCH --error=$SCRATCH/QuGAN/logs/slurm-%j.out
#SBATCH --mail-user={jcw076@latech.edu}
#SBATCH --mail-type=ALL

cd $SCRATCH/QuGAN/Scripts
conda activate myenv

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO

srun --unbuffered --export=ALL bash -c '

	python QGan-Train.py
	python ClassGan-Train.py
'
