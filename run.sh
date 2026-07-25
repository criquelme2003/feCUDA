#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=fecuda
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs.%x.%j

source /usr/local/miniconda3/bin/activate
conda activate fecuda
python test2.py

