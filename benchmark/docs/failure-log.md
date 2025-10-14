# Intentos fallidos y lecciones

## 2025-02-14 – Kernel fusionado sin cola compartida
- **Síntoma:** alto número de atómicas globales y baja ocupación.
- **Causa raíz:** cada hilo escribía directamente en memoria global tras el filtro.
- **Lección:** introducir acumulación en shared memory para minimizar colisiones y reservar espacio contiguo.
- **Estado:** reemplazado por ADR-0001.
