# Efectos Olvidados – CUDA

## Descripción
Implementación experimental para detectar “efectos olvidados” sobre tensores cuadrados usando kernels CUDA. El flujo principal consta de la replicación de los datos mediante bootstraping, para luego aplicar convoluciones maxmin de manera iterativa sobre los datos, permitiendo así identificar efectos olvidados y armar los caminos hacia estos efectos
## Requisitos
- NVIDIA Driver: ≥ 535.xx
- CUDA Toolkit: 13.0 (ver `CMakeLists.txt`)
- GPU recomendada: Ampere o superior (SM 80+)
- Compilador host: GCC 12

## Compilación
```bash
cmake . 
cmake --build build -j
```

## Baseline rápido
Una vez generado el binario principal:
```bash
./build/fecuda_main 
```

## Metodología de medición de rendimiento

Para medir el rendimiento se opta por un tensor de 10x16x16, aplicando bootstraping x10,x100,x1000 y x10000, midiendo el tiempo y memoria (CPU y GPU ) utilizados. 

## Validación de Resultados

Para comprobar el correcto funcionamiento, después de cada cambio dentro del flujo principal, se comprueba el correcto funcionamiento del mismo, comprobando que compile y ejecute sin errores  y validadando los resultados de salida. La validación del output del programa es de dos maneras. 

1. Resultados usando tensores puros (sin bootstrap),comparando con resultados correctos en /validation/data .

2. Resultados usando el bootstraping de 4 etapas mencionado anteriormente (busqueda de valores fuera de rango)
