#!/bin/bash

# Script para generar reportes combinados
# Uso: ./report.sh [opciones]
# Opciones:
#   -i, --input-dir: Directorio de inputs (por defecto: docs)
#   -o, --output: Archivo de reporte final (por defecto: reporte_completo.md)
#   -t, --type: Tipo de reporte (full, summary) (por defecto: full)

set -e

INPUT_DIR="docs"
OUTPUT="reporte_completo.md"
TYPE="full"

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    -t|--type)
      TYPE="$2"
      shift 2
      ;;
    *)
      echo "Opción desconocida: $1"
      echo "Uso: $0 [-i|--input-dir DIR] [-o|--output FILE] [-t|--type TYPE]"
      exit 1
      ;;
  esac
done

echo "Generando reporte $TYPE..."

{
  echo "# Reporte Completo del Proyecto feCUDA"
  echo ""
  echo "Generado el: $(date)"
  echo ""

  if [ -f "$INPUT_DIR/sanitizer_memcheck.txt" ]; then
    echo "## Sanitización - Memcheck"
    echo "\`\`\`"
    head -20 "$INPUT_DIR/sanitizer_memcheck.txt"
    echo "\`\`\`"
    echo ""
  fi

  if [ -f "$INPUT_DIR/benchmark_results.txt" ]; then
    echo "## Benchmarking"
    echo "\`\`\`"
    cat "$INPUT_DIR/benchmark_results.txt"
    echo "\`\`\`"
    echo ""
  fi

  if [ -f "$INPUT_DIR/profile_ncu.ncu-rep" ]; then
    echo "## Profiling NCU"
    echo "Archivo: profile_ncu.ncu-rep"
    echo ""
  fi

  if [ -f "$INPUT_DIR/profile_stats.txt" ]; then
    echo "## Estadísticas NSYS"
    echo "\`\`\`"
    head -30 "$INPUT_DIR/profile_stats.txt"
    echo "\`\`\`"
    echo ""
  fi

} > "$OUTPUT"

echo "Reporte generado en $OUTPUT"