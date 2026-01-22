#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "core/types.cuh"
#include <vector>
#include <cuda_fp16.h>

// Funciones exportadas desde archivos .cu

//------------------------------------------------
template <typename T>
std::vector<std::tuple<int4 *, T *>> maxmin(TensorResult<T> &tensor1, TensorResult<T> &tensor2, T thr, int order);

//------------------------------------------------




#endif
