# Efectos Olvidados – CUDA

## Descripción
Implementación experimental para detectar “efectos olvidados” sobre tensores cuadrados usando kernels CUDA. El flujo principal combina operaciones de maxmin iterativas, cálculo de `prima` y armado de caminos. El objetivo actual es optimizar el kernel fusionado y la cadena de post-procesamiento, conservando verificaciones reproducibles.

## Requisitos
- NVIDIA Driver: ≥ 535.xx
- CUDA Toolkit: 13.0 (ver `CMakeLists.txt`)
- GPU recomendada: Ampere o superior (SM 80+)
- Compilador host: GCC 12

## Compilación
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Baseline rápido
Una vez generado el binario principal:
```bash
./build/fecuda_main --help
```

Para benchmarks reproducibles revisar la carpeta `benchmark/benchmarks` y ejecutar los scripts documentados allí.

## Reproducibilidad
- `benchmark/PERF.md`: metodología estable de medición.
- `benchmark/docs/experiments/*`: registros por experimento.
- `benchmark/docs/ADR/*`: decisiones de arquitectura.
- `benchmark/docs/failure-log.md`: intentos sin éxito y aprendizajes.

## Contribución
Sigue Conventional Commits (`feat:`, `perf:`, `docs:`, etc.) y versionado SemVer. Usa ramas `feat/*`, `perf/*` o `fix/*` según corresponda. Antes de fusionar a `main`, valida la checklist en `benchmark/PERF.md`.
