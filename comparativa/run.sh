#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=fecuda
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs.%x.%j

source /opt/miniconda3/bin/activate
conda activate fecuda
python memory_bounds.py

