#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <core/types.cuh>
#include <kernels/maxmin_kernels.cuh>
#include <utils.cuh>
#include <cuda_fp16.h>

/*

TODO: LIMITAR GRID SIZE Y ADAPTAR KERNEL PARA RECORRER MÁS DE UN K. 
*/

// Versión mejorada de maxmin que usa TensorResult<> y retorna tanto max como min

long long ceil_div_128(__int128 n, long long d) {
    return (long long)((n + d - 1) / d);
}

template <typename T>
void maxmin(const TensorResult<T> &tensor1, const TensorResult<T> &tensor2,
            TensorResult<T> &max_result, TensorResult<T> &min_result, cudaEvent_t &start, cudaEvent_t &end,
            bool keep_in_device)
{

    // Validar que los tensores sean 3D (K=1) como espera el kernel
    if (tensor1.K != 1 || tensor2.K != 1)
    {
        printf("Error: maxmin solo acepta tensores 3D (K=1)\n");
        exit(0); // tensor nulo
    }

    // Para el kernel, necesitamos que A sea [batch, M, K] y B sea [batch, K, N]
    // Pero como K=1, efectivamente son [batch, M] y [batch, N]
    int batch = tensor1.batch;
    int M = tensor1.M;
    int K = tensor1.N; // En el contexto del kernel, N del tensor1 es K
    int N = tensor2.N;

    // Tamaños de memoria
    size_t size_C_min = batch * M * N * K * sizeof(T);
    size_t size_C_max = batch * M * N * sizeof(T);
    
    // Alocar memoria en device
    T *d_A, *d_B, *d_C_min, *d_C_max;
    CHECK_CUDA(cudaMalloc(&d_C_min, size_C_min));
    CHECK_CUDA(cudaMalloc(&d_C_max, size_C_max));
    
    
    if (tensor1.is_device_ptr)
    {
        d_A = tensor1.data;
    }else{
        size_t size_A = batch * M * K * sizeof(T);
        CHECK_CUDA(cudaMalloc(&d_A, size_A));
        CHECK_CUDA(cudaMemcpy(d_A,tensor1.data, size_A, cudaMemcpyHostToDevice));  
    }

    if (tensor2.is_device_ptr)
    {
        d_B = tensor2.data;
    }else{
        size_t size_B = batch * K * N * sizeof(T);
        CHECK_CUDA(cudaMalloc(&d_B, size_B));
        CHECK_CUDA(cudaMemcpy(d_B, tensor2.data, size_B, cudaMemcpyHostToDevice));
    }


    // Copiar datos al device

    constexpr int WARPS_PER_BLOCK = 4;
    int blockSize = WARPS_PER_BLOCK * 32;
    int k_launch_size = (static_cast<int>((K + 31) / 32)) * 32;
    


    __int128 total =
        (__int128)M * (__int128)N * (__int128)batch * (__int128)k_launch_size;

    long long grid_ll = ceil_div_128(total, blockSize);
    
    
    printf("total elements: %lld\n", total);
    printf("grid_ll: %lld\n", grid_ll);

    if (grid_ll > INT_MAX) {
        printf("grid too large: %lld\n", (long long)grid_ll);
        exit(1);
    }
    
    int gridSize = (int)grid_ll;
    std::cout << "dims: (" << gridSize
              << ", " << blockSize
              << ", ksize: " << k_launch_size
              << ")" << std::endl;
    // Ejecutar kernel
    
    CHECK_CUDA(cudaEventRecord(start));
    cub_max_min_kernel<T,WARPS_PER_BLOCK>
        <<<gridSize, blockSize>>>(d_A, d_B, d_C_min, d_C_max, M, K, N, batch);
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_KERNEL()


    // Crear tensor resultado para C_max (solo retornamos C_max para validación)
    T *h_C_max = (T *)malloc(size_C_max);
    T *h_C_min = (T *)malloc(size_C_min); // Si se quiere retornar min también

    CHECK_CUDA(cudaMemcpy(h_C_max, d_C_max, size_C_max, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_min, d_C_min, size_C_min, cudaMemcpyDeviceToHost));

    max_result = TensorResult<T>(h_C_max, false, batch, M, N, 1, true);
    min_result = TensorResult<T>(h_C_min, false, batch, M, N, K, true);
    // Limpiar memoria
    CHECK_CUDA(cudaFree(d_C_min));
    CHECK_CUDA(cudaFree(d_C_max));

    return;
}


template void maxmin<float>(const TensorResult<float> &, const TensorResult<float> &, TensorResult<float> &, TensorResult<float> &, cudaEvent_t&, cudaEvent_t&, bool);

template void maxmin<__half>(const TensorResult<__half> &, const TensorResult<__half> &, TensorResult<__half> &, TensorResult<__half> &, cudaEvent_t&, cudaEvent_t&, bool);