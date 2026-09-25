#include "../../include/algorithms/assemble_paths.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "headers.cuh"
#include "utils.cuh"

// Definición única de assemble_paths, compartida por maxmin v1/v2/v3.
// Ver documentación en include/algorithms/assemble_paths.cuh.
PathsAndValues assemble_paths(
    std::vector<std::vector<int>> prev_paths,
    __half *d_A,
    __half *d_C,
    int *d_argmax,
    float thr,
    int M,
    int N,
    int B,
    int Mpad,
    int Npad,
    int Kpad,
    int *pivot_counter
) {
    std::vector<std::vector<int>> paths;
    std::vector<float> values;

    int total = M * N * B;
    std::vector<__half> h_A(total);
    std::vector<__half> h_C(total);
    std::vector<int> h_argmax(total);

    cudaMemcpy(h_A.data(), d_A, total * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C.data(), d_C, total * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_argmax.data(), d_argmax, total * sizeof(int), cudaMemcpyDeviceToHost);

    auto idx = [&](int b, int m, int n) { return b * Mpad * Npad + m * Npad + n; };


    int blockDim = PIVOT_BLOCKSIZE * PIVOT_BLOCKSIZE;
    size_t free_vram = get_max_available_vram_bytes(1);

    // definir el máximo de pivots por (m,n) a partir de la vram disponible

    int max_pivot = static_cast<int>(free_vram / ((Mpad * Npad) * 3));

    dim3 grid(CEIL_DIV(Mpad, PIVOT_BLOCKSIZE), CEIL_DIV(Npad, PIVOT_BLOCKSIZE), B);
    int shared_mem_size = PIVOT_BLOCKSIZE * PIVOT_BLOCKSIZE * 20 * 3;
    // int *pivot_counter;
    // cudaMalloc(&pivot_counter, sizeof(int));
    
    if (!prev_paths.empty()) {
      
    } else {
      
    }
    return {paths, values};
}
