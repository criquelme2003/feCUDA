#!/bin/bash

# Script para benchmarking del programa
# Uso: ./benchmark.sh [opciones]
# Opciones:
#   -b, --binary: Binario a benchmarkear (por defecto: fecuda_main)
#   -r, --runs: Número de ejecuciones (por defecto: 5)
#   -o, --output: Archivo de salida (por defecto: benchmark_results.txt)

set -e

BINARY="fecuda_main"
BUILD_DIR="build"
RUNS=5
OUTPUT="benchmark_results.txt"

while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--binary)
      BINARY="$2"
      shift 2
      ;;
    -r|--runs)
      RUNS="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Opción desconocida: $1"
      echo "Uso: $0 [-b|--binary BINARIO] [-r|--runs N] [-o|--output FILE]"
      exit 1
      ;;
  esac
done

if [ ! -f "$BUILD_DIR/$BINARY" ]; then
  echo "Error: Binario $BINARY no encontrado en $BUILD_DIR."
  exit 1
fi

echo "Ejecutando benchmark con $RUNS corridas..."
cd "$BUILD_DIR"

{
  echo "Benchmark Results for $BINARY"
  echo "Runs: $RUNS"
  echo "Date: $(date)"
  echo ""
} > "../docs/$OUTPUT"

for i in $(seq 1 $RUNS); do
  echo "Run $i:"
  time ./$BINARY >> "../docs/$OUTPUT" 2>&1
  echo "" >> "../docs/$OUTPUT"
done

echo "Benchmark completado. Resultados en docs/$OUTPUT"