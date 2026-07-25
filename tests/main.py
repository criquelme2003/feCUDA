import numpy as np
import forgethreads as ft
import time
import gc
import ctypes


# Cargar CUDA runtime
cuda = ctypes.CDLL("libcudart.so")

arr = np.load("CC.npy")
print(arr.shape)

tf_a = tf.constant(arr, dtype=tf.float16)
tf_b = tf.constant(arr, dtype=tf.float16)

# ----------------------
# Warmup
# ----------------------
print("Warmup...")
paths, values = ft.maxmin(tf_a, tf_b, 0.5, 1)
cuda.cudaDeviceSynchronize()

# ----------------------
# Benchmark
# ----------------------
thresholds = np.linspace(0.3, 0.7, 5)

times = []
i = 0

for thr in thresholds:
    for _ in range(2):  # 2 ejecuciones por threshold -> 10 total
        
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        current_memory_mib = memory_info['current'] / (1024**2)
        print(f"[Iter {i}] GPU memory: {current_memory_mib:.2f} MiB")

        start = time.perf_counter()

        paths, values = ft.maxmin(tf_a, tf_b, thr, 1)

        paths_tf = tf.experimental.dlpack.from_dlpack(paths)
        values_tf = tf.experimental.dlpack.from_dlpack(values)

        # sincronizar GPU para medir tiempo real
        cuda.cudaDeviceSynchronize()

        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        print(f"thr={thr:.2f} | valores encontrados: {values_tf.shape[0]} | tiempo: {elapsed:.6f}s")
        print(f"--- Iteración {i} completada ---\n")

        # limpiar referencias
        del paths, values, paths_tf, values_tf
        gc.collect()

        i += 1

# ----------------------
# Resultado final
# ----------------------
avg_time = sum(times) / len(times)

print("\n==============================")
print(f"Promedio sobre {len(times)} ejecuciones: {avg_time:.6f} s")
print("==============================")