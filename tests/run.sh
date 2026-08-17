#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=fecuda
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=../logs/logs.%x.%j

../build/fecuda_main