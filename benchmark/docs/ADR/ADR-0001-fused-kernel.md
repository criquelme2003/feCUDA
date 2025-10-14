# ADR-0001: Kernel fusionado maxmin + prima + índices

## Contexto
Se busca reducir accesos a memoria global y sincronizaciones entre kernels al combinar las etapas `maxmin`, cálculo de `prima` y filtrado de índices. El patrón de lanzamiento actual asigna cada bloque a un triplete `(batch, M, N)` y los hilos recorren `K`.

## Decisión
Adoptar un kernel fusionado que:
- Realiza la reducción de maxmin en shared memory.
- Calcula `prima` con datos residentes para evaluar el threshold.
- Emplea una cola intra-bloque antes de escribir en memoria global.

## Consecuencias
**Positivas**
- Menos tráfico global (una sola escritura final).
- Eliminación de kernels intermedios y copias host/device redundantes.
- Base clara para pipelining por etapas dentro del bloque.

**Negativas**
- Kernel más complejo de mantener y perfilar.
- Necesidad de dimensionar shared memory para almacenar colas e índices.

## Estado
Aceptado – 2025-02-14
