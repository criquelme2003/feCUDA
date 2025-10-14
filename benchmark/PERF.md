# Metodología de rendimiento

## KPIs
- Tiempo medio por lanzamiento de kernel (`ms`) + desviación estándar.
- Throughput efectivo: GFLOP/s y GB/s (lectura/escritura DRAM).
- Operational Intensity (FLOPs/byte) para análisis Roofline.
- SM Busy %, Occupancy efectiva, Warp Execution Efficiency.
- Tasa de cache L2 y porcentaje de replay/branch stalls.

## Reglas de medición
1. Compilar en `Release` con LTO y banderas de rendimiento activas.
2. Ejecutar **warm-up ≥ 5** iteraciones antes de medir.
3. Realizar ≥ 30 repeticiones; reportar media, p50, p95, p99 y un IC 95%.
4. Cubrir tamaños pequeños/medios/grandes y casos límite de shared memory o grid.
5. Fijar entorno (clocks, power) cuando se busquen comparaciones exactas.
6. Registrar `git rev`, hash de datos y semillas utilizadas.

## Perfilado (Nsight)
- **Nsight Compute**:
  ```bash
  ncu --set full --target-processes all ./build/fecuda_main
  ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
               smsp__sass_average_branch_targets_threads_uniform.pct ./build/fecuda_main
  ncu --nvtx --nvtx-include "EFO*" --profile-from-start off ./build/fecuda_main
  ncu --launch-skip 10 --launch-count 1 ./build/fecuda_main
  ```
- **Nsight Systems**:
  ```bash
  nsys profile -o benchmark/docs/experiments/out --stats=true ./build/fecuda_main
  ```

## Reportes
- Guardar resultados crudos (CSV/JSON) en `benchmark/docs/experiments/.../results/`.
- Generar tablas agregadas (media, p95, IC 95%) y gráficos boxplot/Roofline.
- Documentar cada cambio relevante en `benchmark/CHANGELOG.md` y, si procede, crear un ADR.

## Checklist previa a merge
- [ ] EXPERIMENT.md completo por cambio significativo.
- [ ] CSV/JSON crudo + tabla agregada versionada.
- [ ] Captura de entorno con `benchmark/scripts/capture_env.sh`.
- [ ] Perfil (`.ncu-rep` o `.nsys-rep`) adjunto al experimento.
- [ ] CHANGELOG y ADR actualizados si aplica.
- [ ] README refleja el estado del baseline.
