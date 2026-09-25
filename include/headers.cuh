#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "core/types.cuh"
#include <vector>
#include <cuda_fp16.h>

#define BLOCKSIZE 32
#define PIVOT_BLOCKSIZE 8

#define MAX_P 20 // Máximo de pivots por (m,n) 

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define Tmax 128
#define MAX_SHARED_MEMORY 48000

struct MaxminResult {
    std::vector<std::vector<std::vector<int>>> paths;  // paths[s] = paths del step s
    std::vector<std::vector<float>>            values; // values[s] = valores del step s
    std::vector<int> effects_per_order; // effects_per_order[s] = nº de efectos (counter) del orden s+1
    int effective_order;
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

// Variante tiled BM×BN con 1D block-tiling (TM resultados por hilo, kernel v3).
MaxminResult maxminv3(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order
);

// Variante tiled BM×BN con 2D block-tiling (TM×TN resultados por hilo, kernel v4).
MaxminResult maxminv4(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order
);

// Variante con conteo de pivots. Si dump_dir no es nullptr, persiste a disco
// (binario crudo + manifest.json) los CSR de testigos de cada orden con
// efectos, para reconstrucción de caminos on-demand posterior.
MaxminResult maxminv2_count(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order,
    const char *dump_dir = nullptr
);

void try_cuco();

#endif
