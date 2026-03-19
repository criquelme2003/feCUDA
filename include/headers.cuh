#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "core/types.cuh"
#include <vector>
#include <cuda_fp16.h>

// Funciones exportadas desde archivos .cu

//------------------------------------------------
// Retorna vector de tuplas (flat_paths, values, count, path_width, effective_order)
//   path_width      = order + 3  =>  cada fila es (b, m, k1, ..., kp, n)
//   effective_order = último orden en que aparecieron caminos nuevos
//                     (puede ser < order si el producto converge antes)
// Para order=1 los paths son cast de int4* (layout idéntico en memoria).
std::vector<std::tuple<int*, __half*, int, int, int>> maxmin(
    TensorResult<__half> &tensor1, TensorResult<__half> &tensor2,
    __half thr, int order);

//------------------------------------------------




#endif
