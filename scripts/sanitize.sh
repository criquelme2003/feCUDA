#!/bin/bash

# Script para ejecutar sanitización con compute-sanitizer
# Uso: ./sanitize.sh [opciones]
# Opciones:
#   -t, --tool: Herramienta (memcheck, racecheck, initcheck, synccheck) (por defecto: memcheck)
#   -b, --binary: Binario a sanitizar (por defecto: fecuda_main)
#   -l, --leak-check: Para memcheck, activar leak-check (full)
#   -o, --output: Archivo de salida (por defecto: sanitizer_<tool>.txt)

set -e

TOOL="memcheck"
BINARY="fecuda_main"
BUILD_DIR="build"
LEAK_CHECK=""
OUTPUT="sanitizer_${TOOL}.txt"

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
    -l|--leak-check)
      LEAK_CHECK="--leak-check=full"
      shift
      ;;
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Opción desconocida: $1"
      echo "Uso: $0 [-t|--tool TOOL] [-b|--binary BINARIO] [-l|--leak-check] [-o|--output FILE]"
      exit 1
      ;;
  esac
done

if [ ! -f "$BUILD_DIR/$BINARY" ]; then
  echo "Error: Binario $BINARY no encontrado en $BUILD_DIR."
  exit 1
fi

echo "Ejecutando compute-sanitizer --tool $TOOL $LEAK_CHECK ./$BINARY > $OUTPUT"
cd "$BUILD_DIR"
compute-sanitizer --tool "$TOOL" $LEAK_CHECK ./$BINARY > "../docs/$OUTPUT"

echo "Sanitización completada. Resultados en docs/$OUTPUT"