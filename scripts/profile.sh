#!/bin/bash

# Script para profiling CUDA con ncu y nsys
# Uso: ./profile.sh [opciones]
# Opciones:
#   -t, --tool: Herramienta (ncu, nsys) (por defecto: ncu)
#   -b, --binary: Binario a perfilar (por defecto: fecuda_main)
#   -m, --metrics: Métricas para ncu (por defecto: algunas clave)
#   -o, --output: Prefijo de salida (por defecto: profile)

set -e

source /etc/profile
module load cuda/12.5

TOOL="ncu"
BINARY="fecuda_main"
BUILD_DIR="build"
METRICS="l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,sm__warps_active.avg.pct_of_peak_sustained_active"
OUTPUT_PREFIX="profile"

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--tool)
      TOOL="$2"
      shift 2
      ;;
    -b|--binary)
      BINARY="$2"
      shift 2
      ;;
    -m|--metrics)
      METRICS="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    *)
      echo "Opción desconocida: $1"
      echo "Uso: $0 [-t|--tool TOOL] [-b|--binary BINARIO] [-m|--metrics METRICS] [-o|--output PREFIX]"
      exit 1
      ;;
  esac
done

if [ ! -f "$BUILD_DIR/$BINARY" ]; then
  echo "Error: Binario $BINARY no encontrado en $BUILD_DIR."
  exit 1
fi

cd "$BUILD_DIR"

if [ "$TOOL" = "ncu" ]; then
  echo "Ejecutando ncu con métricas: $METRICS"
  ncu  -f -o  "../docs/${OUTPUT_PREFIX}_ncu" ./$BINARY
  echo "Perfil ncu completado. Resultados en docs/${OUTPUT_PREFIX}_ncu.ncu-rep"
elif [ "$TOOL" = "nsys" ]; then
  echo "Ejecutando nsys"
  nsys profile --trace=cuda,nvtx --output "../docs/${OUTPUT_PREFIX}_timeline" ./$BINARY
  nsys stats --force-export=true "../docs/${OUTPUT_PREFIX}_timeline.nsys-rep" > "../docs/${OUTPUT_PREFIX}_stats.txt"
  echo "Perfil nsys completado. Resultados en docs/${OUTPUT_PREFIX}_timeline.nsys-rep y docs/${OUTPUT_PREFIX}_stats.txt"
else
  echo "Herramienta no soportada: $TOOL"
  exit 1
fi