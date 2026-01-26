import numpy as np
import forgethreads as ft
import tensorflow as tf
import tensorflow.experimental.dlpack as tf_dlpack

# a = np.ones((1,  1, 6), dtype=np.float16)
# a = a.__dlpack__()
# # Zero-copy
# print("Numpy")
# a = ft.TensorResult(a)


arr = np.random.random_sample((100,))
arr = arr.reshape((1,10,10))

print(f"TENSOR SHAPE: {arr.shape}")
tf_a = tf.constant(arr, dtype=tf.float16)
print(f"TENSOR DEVICE: {tf_a.device}")
tf_a = tf_dlpack.to_dlpack(tf_a)

print("Tensorflow GPU")
tf_t = ft.TensorResult(tf_a)

[paths, values] = ft.maxmin(tf_t, tf_t, 0.4, 1)

vls = tf_dlpack.from_dlpack(values.__dlpack__())
print(f"VALUES DEVICE: {vls.device}")

print(vls)

# a ya no es owner
# t ahora maneja la memoria
