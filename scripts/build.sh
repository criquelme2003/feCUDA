#!/bin/bash

# Script para compilar el proyecto CUDA con CMake
# Uso: ./build.sh [opciones]
# Opciones:
#   -c, --clean: Limpiar el directorio de build antes de compilar
#   -d, --debug: Compilar en modo debug
#   -r, --release: Compilar en modo release (por defecto)
#   -j, --jobs: Número de jobs para make (por defecto: número de CPUs)

set -e

BUILD_DIR="build"
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
MAKE_JOBS=$(nproc)

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--clean)
      CLEAN=true
      shift
      ;;
    -d|--debug)
      CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug"
      shift
      ;;
    -r|--release)
      CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
      shift
      ;;
    -j|--jobs)
      MAKE_JOBS="$2"
      shift 2
      ;;
    *)
      echo "Opción desconocida: $1"
      echo "Uso: $0 [-c|--clean] [-d|--debug|-r|--release] [-j|--jobs N]"
      exit 1
      ;;
  esac
done

if [ "$CLEAN" = true ]; then
  echo "Limpiando directorio de build..."
  rm -rf "$BUILD_DIR"
fi

echo "Creando directorio de build..."
mkdir -p "$BUILD_DIR"

echo "Ejecutando CMake con argumentos: $CMAKE_ARGS"
cd "$BUILD_DIR"
cmake .. $CMAKE_ARGS

echo "Compilando con make -j$MAKE_JOBS..."
make -j"$MAKE_JOBS"

echo "Compilación completada."