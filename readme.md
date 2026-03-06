# Proyecto FECUDA (Finding Extremal CUDA Paths)

Este repositorio contiene una implementación de alto rendimiento de un algoritmo **max‑min** (o extremal) que ejecuta cálculos en GPU mediante CUDA y se integra con TensorFlow a través de DLpack. El objetivo es proporcionar una pieza de infraestructura que
- permite operar sobre tensores de TensorFlow sin copiar datos de la GPU,
- encapsula los resultados en objetos Python seguros (`TensorResult`),
- y demuestra un patrón de comunicación entre Python/C++/CUDA útil para acelerar aplicaciones de análisis numérico.

---
## 🧩 ¿Qué hace el proyecto?

- Carga matrices desde NumPy y las convierte en tensores de `tf.float16`.
- Llama a una función `ft.maxmin` implementada en CUDA (vía `forgethreads`), que devuelve rutas y valores extremos.
- Usa DLpack para intercambiar datos cero‑copia entre TensorFlow y la librería C++/CUDA.
- Gestiona la memoria de GPU con sincronización explícita y evita dobles liberaciones gracias a la clase `TensorResult`.

El ejemplo principal se encuentra en `main.py` y se repite varias iteraciones para medir uso de memoria.

---
## 🏗 Arquitectura general

La arquitectura se puede consultar en **`arch.drawio`**; a continuación se resume con palabras:

1. **Python**: punto de entrada (`main.py`).
2. **Bindings C++** (`src/bindings.cu`) construyen un módulo Python usando PyBind11.
3. Dentro del binding hay una estructura `TensorResult` que encapsula:
   - un `py::capsule` apuntando a un tensor DLpack,
   - métodos `to_dlpack()` para devolver ese tensor a TensorFlow.
4. **TensorFlow** interacciona con el binding usando **DLpack** como puente de datos.
5. **CUDA / C++**: la lógica de cálculo de `maxmin` está en `src/algorithms` y `src/kernels`. La ejecución permanece en GPU y devuelve buffers DLpack.

> 🖼️ El diagrama de `arch.drawio` muestra las conexiones entre Python, los bindings C++, el intercambio DLpack y TensorFlow. Echa un vistazo al archivo para ver las flechas y las conversiones de datos.

---
## ⚙️ Ejemplos

El script `main.py` sirve como referencia:

```python
import numpy as np
import forgethreads as ft
import tensorflow as tf
import tensorflow.experimental.dlpack as tf_dlpack

arr = np.load("CC.npy")

tf_a = tf.constant(arr, dtype=tf.float16)
tf_b = tf.constant(arr, dtype=tf.float16)

[paths, values] = ft.maxmin(tf_a, tf_b, 0.4, 1)

paths_tf = tf.experimental.dlpack.from_dlpack(paths)
values_tf = tf.experimental.dlpack.from_dlpack(values)

print("Valores encontrados:", values_tf.shape[0])
```

Simplemente reemplaza `CC.npy` por tus propios datos y ajusta los parámetros según el problema.

---
## 📊 Benchmarks

Se ha preparado una [comparativa de benchmarks](https://bench-web-sk1k.vercel.app/) (enlace externo). Los resultados actuales comparan diferentes versiones del algoritmo y muestran el uso de memoria/gigapíxeles por segundo.

> 🧠 Nota: el proyecto investiga además el problema de **doble free de managed memory**. La solución vigente consiste en que `TensorResult` sólo se ocupa de liberar sus datos, mientras que el propietario de Python gestiona el ciclo de vida del DLpack.

---
## 🎥 Demo

1. Construye el proyecto con CMake (`mkdir build && cd build && cmake .. && make`).
2. Genera algunos ficheros NumPy de prueba, por ejemplo `np.random.rand(1000,1000).astype('float16')`.
3. Ejecuta `python3 main.py` y observa cómo se imprime el uso de memoria de GPU y el conteo de valores.
4. Puedes abrir `arch.drawio` en [diagrams.net](https://app.diagrams.net/) para visualizar la arquitectura.

Para una demostración rápida en vídeo, se recomienda grabar la salida de la consola durante varias iteraciones mientras se monitorea `nvidia-smi`.

---
## 📁 Estructura relevante

- `src/` – código CUDA/C++.
- `include/` – cabeceras comunes.
- `main.py` – script Python de ejemplo.
- `arch.drawio` – diagrama de arquitectura.
- `requirements.txt` – dependencias Python.

---
## 📝 Licencia y contacto

Añade aquí la información de licencia y cómo contactar al autor o reportar issues.


