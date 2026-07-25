#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "core/types.cuh"
#include <vector>
#include <cuda_fp16.h>

#define BLOCKSIZE 32

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

struct MaxminResult {
    std::vector<std::vector<std::vector<int>>> paths;  // paths[s] = paths del step s
    std::vector<std::vector<float>>            values; // values[s] = valores del step s
    std::vector<int> effects_per_order; // effects_per_order[s] = nº de efectos (counter) del orden s+1
    int                                        effective_order;
};

MaxminResult maxmin(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order
);

// Variante tiled 32×32 con padding (kernel v2).
MaxminResult maxminv2(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order
);

#endif
