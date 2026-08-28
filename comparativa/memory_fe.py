from sparse_generator import sparse_supercritical_block_matrix_julius as generator
import os
from forgeffects_modules.iterative_maxmin_cuadrado import iterative_maxmin_cuadrado
import tensorflow as tf
import tensorflow.errors as tf_errors
import numpy as np
import sys
SEED = 101


# ----- PARSER --------
dimension = sys.argv[1]

def test_dim(dim):
  print(f"testing {dim} dim")
  try:
    n = dim//2
    m,_,_ = generator(n,n,1,SEED)
    eff_order = iterative_maxmin_cuadrado(m, 0.5, 3)
    sys.exit(0)
  except tf_errors.OpError as error:
    print(f"Error {error}")
    sys.exit(1)
  except Exception as error:
    print(f"Error {error}")
    sys.exit(2)
  finally:
    tf.keras.backend.clear_session()

if __name__ == "__main__":
  test_dim(int(dimension))