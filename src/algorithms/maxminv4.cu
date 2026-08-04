#include "../../include/algorithms/assemble_paths.cuh"
#include "../../include/core/types.cuh"
#include "../../include/headers.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <vector>

// 

#define MIN_DIFF 0.01f

#define MAX_GRID_SIZE 10000
#define MAX_PATHS_PER_ITER 100000

#define TM 8
#define TN 8  

#define BK 8
#define BM 64
#define BN 64


// Comprueba que una futura allocation no exceda 2GB
// Comprueba que una futura allocation no exceda 2GB y deje 1GB libre en VRAM

static inline bool check_alloc_size_or_fail(size_t bytes, const char *name) {
    const size_t MAX_ALLOC = 3.5 * 1024 * 1024 * 1024ULL; // 3.5GB
    const size_t RESERVED = 500 * 1024 * 1024ULL;         // 500MB

    if (bytes > MAX_ALLOC) {
        fprintf(stderr, "ERROR: allocation for %s is %zu bytes (>2GB)\n", name, bytes);
        return false;
    }

    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "WARN: cudaMemGetInfo failed (%s). Allowing allocation for %s of %zu bytes\n",
            cudaGetErrorString(err),
            name,
            bytes
        );
        return true; // no info available → be permissive (preserves previous behaviour)
    }

    if (free_bytes < bytes + RESERVED) {
        fprintf(
            stderr,
            "ERROR: not enough free GPU memory for %s: requested=%zubytes free=%zubytes "
            "reserved=%zubytes\n",
            name,
            bytes,
            free_bytes,
            RESERVED
        );
        return false;
    }
    return true;
}

#define CHECK_ALLOC_SIZE_OR_EXIT(bytes, name)                                                      \
    do {                                                                                           \
        if (!check_alloc_size_or_fail((bytes), (name)))                                            \
            exit(EXIT_FAILURE);                                                                    \
    } while (0)

MaxminResult maxminv4(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order
) {

    if (tensor1.getK() != 1 || tensor2.getK() != 1) {
        printf("Error: maxminv2 solo acepta tensores 3D (K=1)\n");
        exit(0);
    }

    // GET NECESARY DIMENSIONS (extents lógicos)
    int B = tensor1.getBatch();
    int M = tensor1.getM();
    int K = tensor1.getN(); // N del tensor1 actúa como K en el kernel
    int N = tensor2.getN();

    // Padea a múltiplo de BM=BN=64 (tamaño de tile del kernel v3). Con tiles de
    // 64, padear sólo a 32 dejaría filas/columnas de borde fuera del buffer
    // físico → OOB en la carga cooperativa. El kernel indexa con strides físicos.
    tensor1.move_to_device(64);
    tensor2.move_to_device(64);
    __half *d_A = (__half *)tensor1.getData();
    __half *d_B = (__half *)tensor2.getData();

    // Dimensiones FÍSICAS (padded). En este flujo K==N ⇒ Kpad==Npad, pero se
    // mantienen separadas por corrección general.
    int Mpad = tensor1.getMPadded();
    int Kpad = tensor1.getNPadded(); // N lógico de tensor1 = K → su padding es Kpad
    int Npad = tensor2.getNPadded();

    // Buffers internos con layout físico [B, Mpad, Npad] (== [B,Mpad,Kpad] aquí).
    size_t padded_elems = (size_t)B * Mpad * Npad;

    // Configuración de lanzamiento (fija para todos los paths)

    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN > 1: iterativo con el mismo kernel
    // ─────────────────────────────────────────────────────────────────────────

    LOG(std::cout << "[MAXMIN C++] M: " << M << "B: " << B << std::endl);

    LOG(std::cout << "[MAXMIN C++] ORDER-" << order << std::endl);

    // Buffers iterativos con layout físico padded: C_dev_before (= C_prev) y
    // C_dev_after (= C_next). Se rellenan con negativo (0xBC) para que el padding
    // no gane el max ni contamine el counter.
    __half *C_dev_before, *C_dev_after;
    int *argmax;
    {
        size_t __alloc_bytes = padded_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_before");
        CHECK_CUDA(cudaMalloc(&C_dev_before, __alloc_bytes));
        CHECK_CUDA(cudaMemset(C_dev_before, 0xBC, __alloc_bytes));
    }
    {
        size_t __alloc_bytes = padded_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_after");
        CHECK_CUDA(cudaMalloc(&C_dev_after, __alloc_bytes));
        CHECK_CUDA(cudaMemset(C_dev_after, 0xBC, __alloc_bytes));
    }

    {
        size_t __alloc_bytes = padded_elems * sizeof(int);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "argmax");
        CHECK_CUDA(cudaMalloc(&argmax, __alloc_bytes));
        CHECK_CUDA(cudaMemset(argmax, -1, __alloc_bytes));
    }
    // C_dev_before arranca como copia física de d_A (ambos [B,Mpad,Kpad]==[B,Mpad,Npad]).
    CHECK_CUDA(
        cudaMemcpy(C_dev_before, d_A, padded_elems * sizeof(__half), cudaMemcpyDeviceToDevice)
    );

    // Número de tiles 32×32 (usar extent lógico; los tiles de borde cubren padding).
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), B);
    float thr_f = __half2float(thr);
    int effective_order = 1;

    int blockDim = ( BM * BN) / (TM * TN);
    dim3 block(blockDim); // define steps for thread loop


    int* d_counter;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));

    std::vector<std::vector<int>> current_paths;
    MaxminResult result;

    for (int s = 0; s < order; s++) {
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

        maxmin_threshold_kernelv4<<<grid, block>>>(
            C_dev_before, d_B, C_dev_after, argmax, d_counter,
            thr, B, M, N, K, Kpad, Npad, -1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        int h_counter = 0;
        CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_counter == 0) {
            LOG(std::cout << "[MAXMIN C++] Convergencia en step " << s + 1 << std::endl);
            break;
        }else{
          LOG(std::cout << "[MAXMIN C++] Efectos encontrados en orden "<< s +1 << " : " << h_counter<< std::endl);
        }

        // Registrar los efectos de este orden (1 entrada por orden con efectos).
        result.effects_per_order.push_back(h_counter);

        effective_order = s + 1;

        // auto [new_paths, new_values] = assemble_paths(
        //     current_paths, C_dev_before, C_dev_after, argmax, thr_f, M, N, B);
        // result.paths.push_back(new_paths);
        // result.values.push_back(new_values);
        // current_paths = std::move(new_paths);

        std::swap(C_dev_before, C_dev_after);
    }

    CHECK_CUDA(cudaFree(d_counter));
    CHECK_CUDA(cudaFree(C_dev_before));
    CHECK_CUDA(cudaFree(C_dev_after));
    CHECK_CUDA(cudaFree(argmax));

    result.effective_order = effective_order;
    return result;
}
