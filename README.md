# FeCUDA – Sprint 2

Implementación CUDA para descubrir efectos iterativos. Tras cerrar el Sprint 1, en este Sprint se reintrodujeron:

- Biblioteca compartida `libeffects_api.so` y un binding C++ → FastAPI básico.
- Servicio REST `/effects` con un frontend mínimo en `services/web`.
- Herramientas de validación alineadas con `fecuda_main --validate`.

## Preparar y compilar

```bash
cmake -S . -B build
cmake --build build
```

La configuración compila sin dependencias opcionales (`mathdx`, `cuTensor`, `cuDNN`) a menos que se activen las opciones `FECUDA_ENABLE_*`.

## API y frontend

1. Levanta el servicio:
   ```bash
   uvicorn services.effects_api:app --host 0.0.0.0 --port 8000
   ```
2. Abre http://localhost:8000/app para usar el visor mínimo (tablas sin gráficos).  
   TODO Sprint 3: reactivar gráficos, métricas avanzadas y descargas (ver `docs/TODO-sprint-3.md`).

## Validación

Ejecuta un ciclo completo (backend + chequeos Python) con:

```bash
tools/run_validation.sh
```

El script lanza `./build/fecuda_main --validate` y luego `python -m validation.validation` (usa `--results-dir` y `--reference-dir` si necesitas rutas personalizadas).

## Notas de sprint

Consulta `SPRINT_NOTES.md` para conocer los alcances incluidos en esta iteración y los faltantes planificados para Sprint 3 (benchmark harness, jobs async, UI avanzada).
