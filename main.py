import numpy as np
import forgethreads as ft
import tensorflow as tf
import tensorflow.experimental.dlpack as tf_dlpack

import gc
import ctypes

# Cargar CUDA runtime una vez
cuda = ctypes.CDLL('libcudart.so')
i = 0
while (i < 10):
    memory_info = tf.config.experimental.get_memory_info('GPU:0')
    current_memory_mib = memory_info['current'] / (1024**2)
    print(f"[Iter {i}] GPU memory: {current_memory_mib:.2f} MiB")

    arr = np.load("CC.npy")

    tf_a = tf.constant(arr, dtype=tf.float16)
    tf_b = tf.constant(arr, dtype=tf.float16)

    [paths, values] = ft.maxmin(tf_a, tf_b, 0.4, 1)

    
    paths_tf = tf.experimental.dlpack.from_dlpack(paths)
    values_tf = tf.experimental.dlpack.from_dlpack(values)

    print(f"Valores encontrados: {values_tf.shape[0]}")
    # ✅ Sincronizar GPU
    cuda.cudaDeviceSynchronize()

    print(f"--- Iteración {i} completada ---\n")
    i += 1

