import forgethreads as ft


from sparse_generator import sparse_supercritical_block_matrix_julius as generator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
    m1,_,_ = generator(n,n,1,SEED)
    m2 = m1.copy()
    eff_order = ft.maxmin(m1,m2, 0.5, 2)
    sys.exit(0)
  except Exception as error:
    print(f"Error {error}")
    sys.exit(1)

if __name__ == "__main__":
  test_dim(int(dimension))

