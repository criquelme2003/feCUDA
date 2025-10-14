---
id: 2025-02-14-exp-001-fused-kernel
autor: Carlos Riquelme
objetivo: Medir impacto del kernel fusionado vs. pipeline anterior
hipotesis: La fusión reduce accesos a global y mejora el tiempo total ≥10%
kpis: [time_ms, GBs, GFLOPs, sm_busy, occupancy]
input: dataset sintético (batch=8, M=N=2048, K=128)
variantes: [baseline-multiple-kernels, fused-kernel]
semillas: [123]
fecha: 2025-02-14
---

## Metodología
- Compilación `Release` con `-lineinfo`.
- Warm-up: 5 iteraciones, repeticiones: 40.
- Captura de entorno con `benchmark/scripts/capture_env.sh results/env.json`.
- ncu: `--nvtx --nvtx-include "EFO*" --set speed-of-light`.

## Resultados
| variante               | media_ms | p95_ms | GB/s | SM_busy% | occ% |
|------------------------|---------:|-------:|-----:|---------:|-----:|
| baseline-multi-kernel  |   TBD    |   TBD  | TBD  |    TBD   | TBD  |
| fused-kernel           |   TBD    |   TBD  | TBD  |    TBD   | TBD  |

## Análisis
- Documentar cuando se dispongan métricas reales.
- Revisar utilidad de NVTX para aislar etapas dentro del kernel fusionado.

## Artefactos
- `results/raw.csv`
- `results/env.json`
- `results/ncu-report.ncu-rep`
