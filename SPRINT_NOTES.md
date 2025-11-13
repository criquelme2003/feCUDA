# Sprint 2 – Reincorporación de servicios

## Lo que se reintrodujo
- `libeffects_api.so` vuelve a construirse desde `services/effects_api_binding.cpp`, limitado a efectos básicos.
- Servicio FastAPI `/effects` y frontend mínimo (`services/web/`) que sólo lista resultados en tablas.
- Script `tools/run_validation.sh` que ejecuta `./build/fecuda_main --validate` antes de correr las verificaciones Python.
- Documentación y README actualizados para reflejar la presencia de API + frontend básicos.

## Lo que sigue pendiente
- Benchmark harness completo y dashboards de comparación.
- Jobs async / colas para procesar solicitudes largas.
- Instrumentación detallada (NVTX, métricas de GPU, exportaciones) y UI avanzada con gráficos.
- Paquetes Docker y pipelines automatizadas.

Los pendientes quedaron documentados en `docs/TODO-sprint-3.md`.
