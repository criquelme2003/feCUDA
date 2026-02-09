#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "core/types.cuh"
#include <vector>
#include <cuda_fp16.h>

// Funciones exportadas desde archivos .cu

//------------------------------------------------
std::vector<std::tuple<int4 *, __half *,int>> maxmin(TensorResult<__half> &tensor1, TensorResult<__half> &tensor2, __half thr, int order);

//------------------------------------------------




#endif
