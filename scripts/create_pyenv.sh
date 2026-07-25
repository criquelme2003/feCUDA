#!/bin/bash

# Script para crear el entorno conda usado para compilar el módulo Python
# (forgethreads) definido en CMakeLists.txt.
#
# CMakeLists.txt fija Python_EXECUTABLE a una ruta absoluta dentro de un
# entorno conda (por defecto "fecuda"). Este script crea ese entorno con
# las versiones de python/pybind11/numpy/tensorflow que CMake y main.py
# esperan encontrar, para que `find_package(pybind11 CONFIG REQUIRED)` y
# `find_package(Python ... COMPONENTS Development)` resuelvan dentro del
# mismo entorno (evita mezclar headers de un entorno con el cmake config
# de otro).
#
# Uso: ./create_env.sh [-n|--name NOMBRE] [-p|--python VERSION] [-f|--force]
# Opciones:
#   -n, --name:   Nombre del entorno conda (por defecto: fecuda)
#   -p, --python: Versión de Python (por defecto: 3.10)
#   -f, --force:  Elimina el entorno si ya existe antes de crearlo

set -e

ENV_NAME="fecuda"
PYTHON_VERSION="3.10"
FORCE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--name)
      ENV_NAME="$2"
      shift 2
      ;;
    -p|--python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    -f|--force)
      FORCE=true
      shift
      ;;
    *)
      echo "Opción desconocida: $1"
      echo "Uso: $0 [-n|--name NOMBRE] [-p|--python VERSION] [-f|--force]"
      exit 1
      ;;
  esac
done

CONDA_BASE=$(conda info --base 2>/dev/null) || {
  echo "Error: no se encontró 'conda' en el PATH. Instala Miniconda/Anaconda primero."
  exit 1
}
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  if [ "$FORCE" = true ]; then
    echo "Eliminando entorno existente '$ENV_NAME'..."
    conda env remove -n "$ENV_NAME" -y
  else
    echo "El entorno '$ENV_NAME' ya existe. Usa -f/--force para recrearlo."
    exit 1
  fi
fi

echo "Creando entorno conda '$ENV_NAME' con Python $PYTHON_VERSION, pybind11, numpy y cmake..."
conda create -n "$ENV_NAME" -y -c conda-forge \
  python="$PYTHON_VERSION" \
  pybind11 \
  numpy \
  cmake

conda activate "$ENV_NAME"

echo "Instalando tensorflow y dependencias de requirements.txt con pip..."
pip install "pip<26"
pip install tensorflow
if [ -f "$(dirname "$0")/../requirements.txt" ]; then
  pip install -r "$(dirname "$0")/../requirements.txt"
fi

ENV_PYTHON="$(conda run -n "$ENV_NAME" which python)"

echo ""
echo "Entorno '$ENV_NAME' creado correctamente."
echo "Python del entorno: $ENV_PYTHON"

CMAKE_PY_LINE=$(grep -n "set(Python_EXECUTABLE" "$(dirname "$0")/../CMakeLists.txt" || true)
if [ -n "$CMAKE_PY_LINE" ] && [[ "$CMAKE_PY_LINE" != *"$ENV_PYTHON"* ]]; then
  echo ""
  echo "AVISO: CMakeLists.txt fija Python_EXECUTABLE a una ruta distinta:"
  echo "  $CMAKE_PY_LINE"
  echo "Actualiza esa línea a:"
  echo "  set(Python_EXECUTABLE $ENV_PYTHON)"
fi

echo ""
echo "Para compilar el proyecto:"
echo "  conda activate $ENV_NAME"
echo "  ./scripts/build.sh -c"
