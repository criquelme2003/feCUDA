#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-env.json}

{
  echo "{"
  echo "  \"datetime\": \"$(date -Is)\","
  echo "  \"git\": \"$(git rev-parse --short HEAD 2>/dev/null || echo n/a)\","
  echo "  \"cuda\": \"$(nvcc --version | tr -s ' ' | tr '\n' ' ' | sed 's/\"//g')\","
  echo "  \"nvidia_smi\": \"$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/\"//g')\""
  echo "}"
} > "$OUT"

echo "Entorno capturado en $OUT"
