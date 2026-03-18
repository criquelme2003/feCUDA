#!/bin/bash

# Script para ejecutar el programa compilado
# Uso: ./run.sh [opciones] [argumentos del programa]
# Opciones:
#   -b, --binary: Especificar el binario a ejecutar (por defecto: fecuda_main)
#   -h, --help: Mostrar ayuda

set -e

BINARY="fecuda_main"
BUILD_DIR="build"

while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--binary)
      BINARY="$2"
      shift 2
      ;;
    -h|--help)
      echo "Uso: $0 [-b|--binary BINARIO] [argumentos del programa]"
      echo "Ejecuta el binario compilado desde el directorio build."
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [ ! -f "$BUILD_DIR/$BINARY" ]; then
  echo "Error: Binario $BINARY no encontrado en $BUILD_DIR. ¿Compilaste el proyecto?"
  exit 1
fi

echo "Ejecutando $BINARY con argumentos: $@"
cd "$BUILD_DIR"
./"$BINARY" "$@"