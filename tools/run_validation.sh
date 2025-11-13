#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$ROOT_DIR/build/fecuda_main"

if [[ ! -x "$BINARY" ]]; then
  echo "[run-validation] No se encontró $BINARY. Ejecuta 'cmake -S . -B build && cmake --build build' primero."
  exit 1
fi

echo "[run-validation] Ejecutando fecuda_main --validate $*"
"$BINARY" --validate "$@"

echo "[run-validation] Validando salidas con validation/validation.py"
python -m validation.validation "$@"
