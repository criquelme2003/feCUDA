#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=bench_etapa3
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs.%x.%j

# Mediciones pendientes de la Etapa 3 (v2 vs v4 y tiempo por orden de v4).
# Requiere recompilar el módulo forgethreads, ya que ft.maxmin ahora acepta
# el argumento kernel=. Lanzar desde comparativa/:  sbatch bench_etapa3.sh

source /opt/miniconda3/bin/activate
conda activate fecuda

# El .so nuevo queda en la raíz del repo (LIBRARY_OUTPUT_DIRECTORY).
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
python bench_etapa3.py "$@"
