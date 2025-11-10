---
id: 2025-02-14-exp-001-fused-kernel
autor: Carlos Riquelme
objetivo: Medir impacto del kernel fusionado (maxmin+prima+índices) vs los kernels separados (uno para cada operación)
hipotesis: La fusión reduce accesos a global y mejora el tiempo total ≥?%
kpis: [time_ms, p95_ms, GBs, GFLOPs, sm_busy, occupancy]
input: Estándar definido para medición de rendimiento
variantes: [baseline-multiple-kernels, fused-kernel]
semillas: ---
fecha: 2025-02-14
---

## Metodología
- Compilación `Release` con `-lineinfo`.
- Warm-up: 5 iteraciones, repeticiones: 40.
- Captura de entorno con `benchmark/scripts/capture_env.sh results/env.json`.
- `benchmark/scripts/profile_fecuda.sh benchmark/docs/experiments/2025-02-14-exp-001-fused-kernel/results ./build/fecuda_main`.
- NVTX: marcar etapas `maxmin`, `prima`, `indices` y `kernel_unificado` en `src/algorithms/maxmin.cu`.
- El script genera automáticamente `ncu-*.csv`, `.ncu-rep`, y reportes `nsys-*`.

## Resultados
| variante               | media_ms | p95_ms | GB/s | SM_busy% | occ% |
|------------------------|---------:|-------:|-----:|---------:|-----:|
| baseline-multi-kernel  |   TODO   |   TODO | TODO |    TODO  | TODO |
| fused-kernel           |   TODO   |   TODO | TODO |    TODO  | TODO |

## Análisis
- A completar tras capturar métricas.
- Confirmar si la reducción de accesos globales se refleja en GB/s y en las métricas de SM busy.
- Si la hipótesis no se cumple, documentar en `docs/failure-log.md`.

## Artefactos
- `results/raw.csv`
- `results/env.json`
- `results/ncu-report.ncu-rep`
- `results/nsys-report.qdrep`
- `results/notes.md` (observaciones manuales)
