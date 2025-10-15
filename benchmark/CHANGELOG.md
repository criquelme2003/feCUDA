# Changelog de rendimiento

## [0.1.0] - 2025-02-14
### Added
- Documentación base de benchmarking (`benchmark/`).
- ADR inicial para el kernel fusionado de maxmin + prima + índices.
- Plantilla de experimentos y captura de entorno.

### Notes
- Versión inicial enfocada en estandarizar medición; no incluye resultados históricos.

## [0.1.1] - 2025-02-14
### Added
- Detalles metodológicos para el experimento `exp-001-fused-kernel`.
- Lista de artefactos esperados (CSV, ncu/nsys, notas) asociada al kernel unificado.

### Changed
- KPIs del experimento para incluir percentiles y throughput explícito.

## [0.1.2] - 2025-02-14
### Added
- Script `benchmark/scripts/profile_fecuda.sh` para generar reportes Nsight Compute y Nsight Systems (más CSV de métricas clave).
- Configuración JSON de variantes (`benchmark/config/benchmarks.json`) y script Python `run_benchmarks.py` para corridas personalizadas (si se desea mantener CSV).

### Changed
- `EXPERIMENT.md` ahora hace referencia directa al script de perfilado.
