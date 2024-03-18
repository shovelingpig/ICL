#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --mem=900GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -o logs/%j.out

srun $1