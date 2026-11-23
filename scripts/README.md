# Scripts Útiles para feCUDA

Esta carpeta contiene scripts modularizados para facilitar el desarrollo, testing y profiling del proyecto feCUDA.

## Scripts Disponibles

### build.sh
Compila el proyecto usando CMake.
- Uso: `./build.sh [-c|--clean] [-d|--debug|-r|--release] [-j|--jobs N]`
- Opciones:
  - `-c`: Limpiar build antes de compilar
  - `-d`: Modo debug
  - `-r`: Modo release (por defecto)
  - `-j`: Número de jobs para make

### run.sh
Ejecuta el binario compilado.
- Uso: `./run.sh [-b|--binary BINARIO] [argumentos]`
- Opciones:
  - `-b`: Especificar binario (por defecto: fecuda_main)

### sanitize.sh
Ejecuta sanitización con compute-sanitizer.
- Uso: `./sanitize.sh [-t|--tool TOOL] [-b|--binary BINARIO] [-l|--leak-check] [-o|--output FILE]`
- Opciones:
  - `-t`: Herramienta (memcheck, racecheck, initcheck, synccheck)
  - `-b`: Binario a sanitizar
  - `-l`: Activar leak-check para memcheck
  - `-o`: Archivo de salida

### profile.sh
Realiza profiling con ncu o nsys.
- Uso: `./profile.sh [-t|--tool TOOL] [-b|--binary BINARIO] [-m|--metrics METRICS] [-o|--output PREFIX]`
- Opciones:
  - `-t`: Herramienta (ncu, nsys)
  - `-b`: Binario
  - `-m`: Métricas para ncu
  - `-o`: Prefijo de salida

### benchmark.sh
Ejecuta benchmarking con múltiples corridas.
- Uso: `./benchmark.sh [-b|--binary BINARIO] [-r|--runs N] [-o|--output FILE]`
- Opciones:
  - `-b`: Binario
  - `-r`: Número de corridas
  - `-o`: Archivo de salida

### report.sh
Genera un reporte combinado de los resultados.
- Uso: `./report.sh [-i|--input-dir DIR] [-o|--output FILE] [-t|--type TYPE]`
- Opciones:
  - `-i`: Directorio de inputs (docs)
  - `-o`: Archivo de salida
  - `-t`: Tipo de reporte (full, summary)

## Flujo Típico de Uso

1. Compilar: `./build.sh`
2. Sanitizar: `./sanitize.sh`
3. Perfilar: `./profile.sh`
4. Benchmarkear: `./benchmark.sh`
5. Generar reporte: `./report.sh`

Todos los outputs se guardan en la carpeta `docs/`.