## Lista de mejoras
### Prompt:
Para cada mejora de este checklist, sigue este flujo:

1. **Baseline**: antes de tocar código, compila el estado actual (`cmake --build build`) y perfila con `scripts/profile.sh` (usa `module load cuda/12.5` internamente, coherente con la toolchain del proyecto). Ejecuta:
   `ncu --set full --export docs/baseline_<nombre_mejora> ./build/fecuda_main`
   y vuelca el resumen legible por consola con:
   `ncu --import docs/baseline_<nombre_mejora>.ncu-rep --page details --print-summary per-kernel`
   Antes de correr esto, revisa `src/main.cu` (función `main`) para saber qué configuraciones (`B,M,thr`) se están lanzando — si hay varias llamadas a `run_and_validate` seguidas, comenta todas menos la configuración objetivo para que el reporte no mezcle kernels/tamaños distintos.
2. **Aplica la mejora** descrita en el ítem del checklist.
3. **Post-cambio**: recompila y repite el mismo comando de profiling, guardando el reporte como `docs/postfix_<nombre_mejora>`.
4. **Compara** al menos estas métricas entre baseline y post-cambio: `Duration [ms]`, `Compute (SM) Throughput [%]`, `Memory Throughput [%]`, `L1/TEX Cache Throughput [%]`, `L2 Cache Throughput [%]`, `DRAM Throughput [%]`, `Achieved Occupancy [%]`.
5. **Documenta el resultado** en una nueva sección de este markdown (debajo del ítem correspondiente) con: título, breve descripción de qué se cambió y por qué, y una tabla baseline vs. post-cambio con las métricas del punto 4. Sé neutral en la conclusión — reporta lo que ocurrió aunque el cambio no haya mejorado el rendimiento (o lo haya empeorado); ese es un resultado válido y útil.


- [x] Con M=N=K=1000, una columna completa de B_mat (K=1000 elementos, 2KB en __half) cabe entera en shared memory (límite ~65KB/bloque). Cargarla completa una sola vez por bloque en vez de tilear en trozos, evitando que cada uno de los M bloques vuelva a pedirla repetidamente a L2.

## Resultado: cachear columna de B_mat en shared memory

**Qué se cambió**: en `maxmin_threshold_kernel` ([src/kernels/maxmin/kernel_v1.cu](src/kernels/maxmin/kernel_v1.cu)), se agregó una carga cooperativa previa a la reducción, donde todos los threads del bloque copian `B_mat[b,:,n]` completa a un buffer `__shared__` (`s_bcol`, tamaño `K * sizeof(__half)`). El bucle de reducción por thread pasó de leer `A_mat[a_idx]` y `B_mat[b_idx]` (2 accesos a memoria global por iteración) a leer `A_mat[a_idx]` y `s_bcol[k]` (1 acceso a global + 1 a shared). El tamaño de shared memory por bloque se ajustó en [src/algorithms/maxmin.cu](src/algorithms/maxmin.cu) sumando `K * sizeof(__half)` al cálculo original.

Configuración de prueba: `B=1, M=N=K=1000, thr=0.5` (única configuración activa en `src/main.cu` durante la medición). Se validó que el conteo de efectos (`h_counter`) coincide exactamente con la referencia CPU en ambos casos (472326), tanto antes como después del cambio.

**Comparación baseline vs. post-cambio:**

| Métrica | Baseline | Post-cambio | Δ |
|---|---|---|---|
| Duration [ms] | 14.58 | 11.39 | **-21.9%** |
| Compute (SM) Throughput [%] | 5.19 | 10.45 | +5.26 pp |
| Memory Throughput [%] | 87.09 | 90.71 | +3.62 pp |
| L1/TEX Cache Throughput [%] | 37.87 | 39.64 | +1.77 pp |
| L2 Cache Throughput [%] | 87.09 | 90.71 | +3.62 pp |
| DRAM Throughput [%] | 0.03 | 0.04 | ~igual |
| Achieved Occupancy [%] | 99.16 | 99.55 | ~igual |
| Registers per Thread | 16 | 26 | +10 |

**Interpretación**: la duración mejoró ~22%, pero no por reducir la presión sobre L2 — de hecho el throughput de L2 *subió* en vez de bajar. La hipótesis inicial (que el cuello de botella eran relecturas redundantes de B_mat entre los M bloques que comparten columna) no se refleja en una caída de L2 Throughput; cada bloque sigue trayendo la columna completa desde memoria global una sola vez, mismo volumen de datos total. La mejora real parece venir de eliminar un acceso a memoria global por iteración dentro del bucle de reducción (quedó solo 1 en vez de 2), lo cual se refleja en el Compute Throughput duplicado (5.19%→10.45%) y en menor tiempo total — el kernel satura L2 en una ventana de tiempo más corta. DRAM y Occupancy se mantienen prácticamente sin cambios, como se esperaba (el working set ya cabía en L2 antes del cambio).

Reportes crudos: `docs/baseline_shared_bmat.ncu-rep`, `docs/postfix_shared_bmat.ncu-rep`.