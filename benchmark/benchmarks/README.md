# Harness de benchmarks

## Objetivo
Centralizar ejecutables y utilidades para medir el rendimiento del módulo de efectos olvidados. Los scripts y binarios aquí descritos deben poder correrse desde `build/` tras compilar en `Release`.

## Componentes
- `timer.cu`: ejemplo mínimo de medición con eventos CUDA.
- `run.sh`: (pendiente) script para lanzar benchmarks y guardar CSV en `docs/experiments/.../results/`.

## Uso rápido
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
nvcc -O3 benchmark/benchmarks/timer.cu -o build/benchmark_timer
./build/benchmark_timer --size 2048 --reps 40
```

## Notas
- Asegúrate de ejecutar `benchmark/scripts/capture_env.sh` antes y después de sesiones largas.
- Etiqueta commits que cambien este harness con el prefijo `bench:`.
