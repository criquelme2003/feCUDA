#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "core/types.cuh"
#include <vector>
#include <cuda_fp16.h>

struct MaxminResult {
    std::vector<std::vector<std::vector<int>>> paths;  // paths[s] = paths del step s
    std::vector<std::vector<float>>            values; // values[s] = valores del step s
    int                                        effective_order;
};

MaxminResult maxmin(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order
);

#endif
