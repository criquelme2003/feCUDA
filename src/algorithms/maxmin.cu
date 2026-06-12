#include "../../include/core/types.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <vector>

bool g_verbose = true; // definición del global (extern en utils.cuh)

#define MAX_GRID_SIZE 10000
#define MAX_PATHS_PER_ITER 100000

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

// ─────────────────────────────────────────────────────────────────────────────
// Prepend step_order a cada fila (b,m,k,n) y acumula en out_host.
// raw4: buffer host con count*4 ints   layout: [b, m, k, n] por fila
// out_host: vector acumulador con path_width=5
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<int>> assemble_paths(
    std::vector<std::vector<int>> prev_paths, __half *d_A, __half *d_C, int M, int N, int B
) {
    std::vector<std::vector<int>> ret;

    std::vector<__half> h_A(M*N*B);
    std::vector<__half> h_C(M*N*B);

    cudaMemcpy(h_A.data(), d_A, M * N * B * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C.data(), d_C, M * N * B * sizeof(__half), cudaMemcpyDeviceToHost);

    if (prev_paths.size() > 0) {
        for (auto path : prev_paths) {
          for(int i = 0; i< h_A)
        }
    } else {
      for()

    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Maxmin retorna d_paths, d_values, h_total_count, path_width, effective_order

std::vector<std::tuple<int *, __half *, int, int, int>> maxmin(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order,
    bool return_paths
) {
    std::vector<std::tuple<int *, __half *, int, int, int>> ret;

    if (tensor1.getK() != 1 || tensor2.getK() != 1) {
        printf("Error: maxmin solo acepta tensores 3D (K=1)\n");
        exit(0);
    }

    // GET NECESARY DIMENSIONS
    int B = tensor1.getBatch();
    int M = tensor1.getM();
    int K = tensor1.getN(); // N del tensor1 actúa como K en el kernel
    int N = tensor2.getN();
    int total_elems = B * M * N;
    tensor1.move_to_device();
    tensor2.move_to_device();
    __half *d_A = (__half *)tensor1.getData();
    __half *d_B = (__half *)tensor2.getData();
    // Configuración de lanzamiento (fija para todos los paths)
    int blockDim = 128;
    dim3 block(blockDim); // define septs for thread loop
    size_t shmem =
        blockDim * (sizeof(__half) +
                    sizeof(int)); // shared memory save k_values and k_index for block reduction

    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN > 1: iterativo con el mismo kernel
    // ─────────────────────────────────────────────────────────────────────────

    LOG(std::cout << "[MAXMIN C++] M: " << M << "B: " << B << std::endl);

    LOG(std::cout << "[MAXMIN C++] ORDER-" << order
                  << " return_paths=" << (return_paths ? "true" : "false") << std::endl);

    // Buffers iterativos: C_dev_before (= C_prev) y C_dev_after (= C_next)
    __half *C_dev_before, *C_dev_after;
    int *argmax;
    {
        size_t __alloc_bytes = (size_t)total_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_before");
        CHECK_CUDA(cudaMalloc(&C_dev_before, __alloc_bytes));
    }
    {
        size_t __alloc_bytes = (size_t)total_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_after");
        CHECK_CUDA(cudaMalloc(&C_dev_after, __alloc_bytes));
    }

    {
        size_t __alloc_bytes = (size_t)total_elems * sizeof(int);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_after");
        CHECK_CUDA(cudaMalloc(&argmax, __alloc_bytes));
        CHECK_CUDA(cudaMemset(argmax, -1, __alloc_bytes));
    }
    // C_dev_before arranca como copia de d_A
    CHECK_CUDA(
        cudaMemcpy(C_dev_before, d_A, total_elems * sizeof(__half), cudaMemcpyDeviceToDevice)
    );

    dim3 grid(N, M, B);
    float thr_f = __half2float(thr);
    int effective_order = 0;

    // ─────────────────────────────────────────────────────────────────────────
    // return_paths = false: emitir aristas crudas por step
    // ─────────────────────────────────────────────────────────────────────────
    if (!return_paths) {

        for (int s = 0; s < order; s++) {

            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                C_dev_before,
                d_B,
                C_dev_after,
                argmax,
                thr,
                B,
                M,
                N,
                K,
                -1
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            effective_order++;
            std::swap(C_dev_before, C_dev_after);
        }

        CHECK_CUDA(cudaFree(C_dev_before));
        CHECK_CUDA(cudaFree(C_dev_after));

        return ret;
    }
}
