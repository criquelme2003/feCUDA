# Sprint 1 Notes

Este documento resume los recortes hechos para cerrar el Sprint 1 y deja claro qué queda pendiente para el siguiente ciclo.

## Módulos fuera de alcance (eliminados de la rama)

- **Kernels experimentales** (`kernel_v2`, variantes fused/lineales, transposes dedicados). Se eliminaron los fuentes y solo queda `src/kernels/maxmin/kernel_v1.cu`.
- **Bootstrap y efectos empíricos** (`algorithms/bootstrap*`, `kernels/bootstrap`, `FEempirical`). Todo el código y los artefactos de bootstrap salieron del árbol hasta que el core estable esté validado.
- **Servicios y frontend** (`services/`, `effects_api`, `frontend/`). No hay API ni UI en esta rama; solo se conserva la ejecución batch local.
- **Benchmarking extendido** (`bench.cu`, scripts de profiling y reportes en `results/benchmarks`). Los proyectos y datos asociados se removieron para dejar un footprint mínimo.

## Funcionalidad activa

- Kernel `maxmin` + `iterative_maxmin_cuadrado`.
- Utilidades mínimas para lectura de datasets (`datasets_txt`), construcción de caminos y almacenamiento de resultados.
- Generación de artefactos de validación (`fecuda_validator`) y script `validation/validation.py`.

## Pendientes para Sprint 2

1. Reintegrar bootstrap y pruebas con réplicas grandes (requiere revisar dependencias `curand` y políticas de memoria).
2. Revisar caminos para exponer una API (gRPC/FastAPI) y retomar `effects_api`.
3. Volver a habilitar los kernels alternativos cuando existan comparaciones de rendimiento y casos de prueba.
4. Completar el CLI/Frontend una vez que el core tenga KPIs de desempeño publicados.

Mientras tanto, cualquier contribución debe enfocarse en robustecer el flujo actual: cargar datasets, ejecutar `iterative_maxmin_cuadrado`, guardar y validar resultados.
