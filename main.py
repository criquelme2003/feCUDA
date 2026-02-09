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

    tf_a_dlpack = tf_dlpack.to_dlpack(tf_a)
    tf_b_dlpack = tf_dlpack.to_dlpack(tf_b)

    tf_ta = ft.TensorResult(tf_a_dlpack)
    tf_tb = ft.TensorResult(tf_b_dlpack)

    [paths, values] = ft.maxmin(tf_ta, tf_tb, 0.4, 1)

    vls = tf_dlpack.from_dlpack(values.__dlpack__())
    pts = tf_dlpack.from_dlpack(paths.__dlpack__())

    print(f"Valores encontrados: {vls.shape[0]}")

    # ✅ Limpiar TODO antes de siguiente iteración
    del vls, pts
    del paths, values
    del tf_ta, tf_tb
    del tf_a, tf_b
    del arr

    # ✅ Sincronizar GPU
    cuda.cudaDeviceSynchronize()

    # ✅ Forzar garbage collection
    gc.collect()

    print(f"--- Iteración {i} completada ---\n")
    i += 1
