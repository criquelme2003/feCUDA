#ifndef PATH_KERNELS_CUH
#define PATH_KERNELS_CUH
#include <cuda_fp16.h>
__global__ void maxmin_all_pivots(
    const __half *__restrict__ A,             // [B,Mpad,Kpad] factor izq. = C_prev
    const __half *__restrict__ B,             // [B,Kpad,Npad] factor der. = B_orig [FIJA]
    __half *__restrict__ C,                   // [B,Kpad,Npad] factor der. = B_orig [FIJA]
    int *__restrict__ pivots,                 // cuenta los pivots por cada C (<=Tmax)
    unsigned long long *__restrict__ counter, // nullable — cuenta celdas con efecto >= thr
    __half thr,
    int numBatches,
    int M,
    int N,
    int K,
    int Kpad,
    int Npad,
    int batch_id
);

// Segunda pasada: escribe el CSR de testigos 
__global__ void maxmin_write_all_pivots(
    const __half *__restrict__ A,
    const __half *__restrict__ B,
    const __half *__restrict__ C,
    const int *__restrict__ pivots_detailed_count,
    const int *__restrict__ acc_pivots_detailed_count,
    int *__restrict__ pivots,
    int numBatches,
    int M,
    int N,
    int K,
    int Kpad,
    int Npad,
    int batch_id
);

#endif
