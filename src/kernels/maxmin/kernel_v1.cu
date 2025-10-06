
#include <float.h>
#include "../../../include/utils/cuda_utils.cuh"
#include "../../../include/core/types.cuh"

__global__ void max_min_kernel(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    float *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{

    extern __shared__ float mins[];

    unsigned k = threadIdx.x;
    unsigned int n = blockIdx.x;
    unsigned int m = blockIdx.y;
    unsigned int b = blockIdx.z;

    unsigned int a_idx = b * M * K + m * K + k;
    unsigned int b_idx = b * K * N + k * N + n;
    unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + k;

    if (k < K)
    {
        mins[k] = fminf(A[a_idx], B[b_idx]);
        C_min[c_min_idx] = mins[k];
    }
    else
    {
        mins[k] = FLT_MAX;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (k < s && (k + s) < K)
        {
            float a = mins[k];
            float b = mins[k + s];
            mins[k] = fmaxf(a, b); // usa fminf si quisieras mínimo final
        }
        __syncthreads();
    }

    if (k == 0)
    {
        C_max[b * M * N + m * N + n] = mins[0];
    }
}

// Función wrapper para validación - ejecuta el kernel v1 y retorna solo C_max
TensorResult maxmin_kernel_v1_wrapper(const TensorResult &tensor_a, const TensorResult &tensor_b)
{

    // Validar que los tensores sean 3D (K=1) como espera el kernel
    if (tensor_a.K != 1 || tensor_b.K != 1)
    {
        printf("Error: maxmin_kernel_v1 solo acepta tensores 3D (K=1)\n");
        return TensorResult(); // tensor nulo
    }

    // Para el kernel, necesitamos que A sea [batch, M, K] y B sea [batch, K, N]
    // Pero como K=1, efectivamente son [batch, M] y [batch, N]
    int batch = tensor_a.batch;
    int M = tensor_a.M;
    int K = tensor_a.N; // En el contexto del kernel, N del tensor_a es K
    int N = tensor_b.N;

    printf("Ejecutando kernel_v1 con dimensiones: batch=%d, M=%d, K=%d, N=%d\n", batch, M, K, N);

    // Tamaños de memoria
    size_t size_A = batch * M * K * sizeof(float);
    size_t size_B = batch * K * N * sizeof(float);
    size_t size_C_min = batch * M * N * K * sizeof(float);
    size_t size_C_max = batch * M * N * sizeof(float);

    // Alocar memoria en device
    float *d_A, *d_B, *d_C_min, *d_C_max;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_min, size_C_min);
    cudaMalloc(&d_C_max, size_C_max);

    // Preparar datos host si es necesario
    float *h_A = tensor_a.data;
    float *h_B = tensor_b.data;
    bool liberar_A = false, liberar_B = false;

    if (tensor_a.is_device_ptr)
    {
        h_A = (float *)malloc(size_A);
        cudaMemcpy(h_A, tensor_a.data, size_A, cudaMemcpyDeviceToHost);
        liberar_A = true;
    }

    if (tensor_b.is_device_ptr)
    {
        h_B = (float *)malloc(size_B);
        cudaMemcpy(h_B, tensor_b.data, size_B, cudaMemcpyDeviceToHost);
        liberar_B = true;
    }

    // Copiar datos al device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Configurar grid y bloques para el kernel
    dim3 blockSize(nextPow2(K)); // la potencia de 2 mas cercana threads por bloque
    dim3 gridSize(N, M, batch);  // Grid de (N, M, batch)
    size_t shared_mem_size = K * sizeof(float);

    // Ejecutar kernel
    max_min_kernel<<<gridSize, blockSize, shared_mem_size>>>(
        d_A, d_B, d_C_min, d_C_max, M, K, N, batch);

    cudaDeviceSynchronize();

    // Verificar errores del kernel
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        printf("Error en kernel_v1: %s\n", cudaGetErrorString(kernel_error));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_min);
        cudaFree(d_C_max);
        if (liberar_A)
            free(h_A);
        if (liberar_B)
            free(h_B);
        return TensorResult();
    }

    // Crear tensor resultado para C_max (solo retornamos C_max para validación)
    float *h_C_max = (float *)malloc(size_C_max);
    cudaMemcpy(h_C_max, d_C_max, size_C_max, cudaMemcpyDeviceToHost);

    TensorResult resultado(h_C_max, false, batch, M, N, 1, true);

    // Limpiar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_min);
    cudaFree(d_C_max);
    if (liberar_A)
        free(h_A);
    if (liberar_B)
        free(h_B);

    printf("Kernel_v1 ejecutado exitosamente\n");
    return resultado;
}
