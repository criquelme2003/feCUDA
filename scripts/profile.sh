#!/bin/bash

# Script para profiling CUDA con ncu y nsys.
# Acopla la salida a la convención de docs/nvidia-reports/rules.md:
#   <kernel>_<input-sizes>.ncu-rep   (ncu)
#   <kernel>_<input-sizes>.nsys-rep  (nsight systems)
#
# Uso: ./profile.sh --kernel <nombre> --sizes <M> [opciones]
# Opciones:
#   -k, --kernel:  nombre lógico del kernel (obligatorio) → prefijo del reporte
#   -s, --sizes:   tamaño de entrada M (obligatorio) → sufijo del reporte y arg del binario
#   -B, --batch:   batch B (por defecto: 1)
#   -T, --thr:     threshold (por defecto: 0.5)
#   -t, --tool:    herramienta ncu | nsys (por defecto: ncu)
#   -b, --binary:  binario a perfilar (por defecto: fecuda_main)
#   -S, --set:     set de secciones de ncu (por defecto: full → todas las secciones
#                  en la ncu-ui). Alternativas: detailed, basic, roofline, …

set -e

source /etc/profile
module load cuda/12.5

TOOL="ncu"
BINARY="fecuda_main"
BUILD_DIR="build"
REPORT_DIR="docs/nvidia-reports"
# Set de secciones de ncu. "full" recolecta todas las secciones (reporte completo
# en la ncu-ui); es lo más lento por el replay múltiple del kernel.
NCU_SET="full"
KERNEL=""
SIZES=""
BATCH="1"
THR="0.5"

while [[ $# -gt 0 ]]; do
  case $1 in
    -k|--kernel)  KERNEL="$2";  shift 2 ;;
    -s|--sizes)   SIZES="$2";   shift 2 ;;
    -B|--batch)   BATCH="$2";   shift 2 ;;
    -T|--thr)     THR="$2";     shift 2 ;;
    -t|--tool)    TOOL="$2";    shift 2 ;;
    -b|--binary)  BINARY="$2";  shift 2 ;;
    -S|--set)     NCU_SET="$2"; shift 2 ;;
    *)
      echo "Opción desconocida: $1"
      echo "Uso: $0 --kernel <nombre> --sizes <M> [-B batch] [-T thr] [-t ncu|nsys] [-b binario] [-S set]"
      exit 1
      ;;
  esac
done

# Validaciones: kernel y sizes son obligatorios para nombrar según la convención.
if [ -z "$KERNEL" ] || [ -z "$SIZES" ]; then
  echo "Error: --kernel y --sizes son obligatorios (convención <kernel>_<input-sizes>)."
  echo "Uso: $0 --kernel <nombre> --sizes <M> [opciones]"
  exit 1
fi

if [ ! -f "$BUILD_DIR/$BINARY" ]; then
  echo "Error: Binario $BINARY no encontrado en $BUILD_DIR."
  exit 1
fi

# Nombre del reporte según docs/nvidia-reports/rules.md: <kernel>_<input-sizes>
REPORT_NAME="${KERNEL}_${SIZES}"
# Ruta absoluta al directorio de reportes (se corre desde BUILD_DIR).
mkdir -p "$REPORT_DIR"
ABS_REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"

# Argumentos que recibe el binario: M B thr (una sola corrida GPU-only).
APP_ARGS="$SIZES $BATCH $THR"

cd "$BUILD_DIR"

if [ "$TOOL" = "ncu" ]; then
  echo "Ejecutando ncu (kernel=$KERNEL sizes=$SIZES B=$BATCH thr=$THR set=$NCU_SET)"
  ncu --set "$NCU_SET" -f -o "${ABS_REPORT_DIR}/${REPORT_NAME}" ./$BINARY $APP_ARGS
  echo "Perfil ncu completado → ${REPORT_DIR}/${REPORT_NAME}.ncu-rep"
elif [ "$TOOL" = "nsys" ]; then
  echo "Ejecutando nsys (kernel=$KERNEL sizes=$SIZES B=$BATCH thr=$THR)"
  nsys profile --trace=cuda,nvtx --output "${ABS_REPORT_DIR}/${REPORT_NAME}" ./$BINARY $APP_ARGS
  nsys stats --force-export=true "${ABS_REPORT_DIR}/${REPORT_NAME}.nsys-rep" \
    > "${ABS_REPORT_DIR}/${REPORT_NAME}_stats.txt"
  echo "Perfil nsys completado → ${REPORT_DIR}/${REPORT_NAME}.nsys-rep (+ _stats.txt)"
else
  echo "Herramienta no soportada: $TOOL"
  exit 1
fi
