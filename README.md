# FeCUDA – Sprint 1 Core

Este repositorio refleja el estado acordado para el cierre del Sprint 1. El alcance se limita al kernel `maxmin`, a `iterative_maxmin_cuadrado` y al flujo mínimo para leer datasets locales, generar caminos y validarlos. No hay servicios, API, frontend ni pipelines de benchmarking activos en esta rama.

## Requisitos

- CUDA Toolkit 11.4+ y un GPU NVIDIA compatible.
- CMake 3.18 o superior.
- Python 3.9+ (solo para correr `validation/validation.py`).

## Build

```bash
cmake -S . -B build
cmake --build build
```

Solo se generan tres artefactos:

- `fecuda_core`: biblioteca estática con los kernels y algoritmos base.
- `fecuda_main`: ejecutable batch local.
- `fecuda_validator`: generador de artefactos para las validaciones.

## Ejecución rápida

1. **Main batch local**
   ```bash
   ./build/fecuda_main
   ```
   - Lee `datasets_txt/CC.txt`.
   - Ejecuta `iterative_maxmin_cuadrado` con `threshold=0.3` y `order=4`.
   - Guarda los caminos resultantes en `results/paths_values_cc_local.json`.

2. **Generación de artefactos de validación**
   ```bash
   ./build/fecuda_validator
   python validation/validation.py
   ```
   - El ejecutable GPU genera `validation/results/paths_values_<dataset>_<thr>.json`.
   - El script compara esas salidas contra `validation/data/`.

Ambos binarios asumen que se ejecutan desde la raíz del repositorio (se incrusta `FECUDA_SOURCE_DIR` en la compilación), por lo que no dependen de rutas relativas frágiles.

## Estructura mínima del repo

- `include/` y `src/` solo contienen el core (`TensorResult`, utilidades básicas y `max_min_kernel`).
- `datasets_txt/` almacena los datasets estáticos usados por `fecuda_main` y el validador.
- `validation/` mantiene los JSON de referencia y el script `validation.py`.
- `results/` queda vacío salvo por los artefactos que genere cada ejecución local.

## Estado del Sprint

- ✅ Núcleo funcional (`maxmin`, `indices`, `armar_caminos`, `iterative_maxmin_cuadrado`).
- ✅ Validaciones básicas sobre datasets estáticos (`CC`, `EE`).
- ❌ Sin bootstrap, servicios, frontend ni kernels experimentales. Estos módulos quedan documentados en `SPRINT_NOTES.md` para reactivarlos en Sprint 2.

Para detalles sobre los módulos excluidos y próximos pasos, revisa `SPRINT_NOTES.md`.
