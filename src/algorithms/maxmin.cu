#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <maxmin_kernels.cuh>
#include <device_launch_parameters.h>
#include <core/types.cuh>
#include <utils.cuh>

#include <headers.cuh>

// Implementación en CPU que replica la lógica del kernel max_min_kernel sin paralelización.
static void maxmin_cpu(const TensorResult &tensor1, const TensorResult &tensor2,
                       TensorResult &max_result, TensorResult &min_result)
{
    max_result = TensorResult();
    min_result = TensorResult();

    if (tensor1.K != 1 || tensor2.K != 1)
    {
        printf("Error: maxmin_kernel_v1 solo acepta tensores 3D (K=1)\n");
        return;
    }

    const int batch = tensor1.batch;
    const int M = tensor1.M;
    const int K = tensor1.N; // tensor1.N actúa como K
    const int N = tensor2.N;

    if (tensor2.batch != batch || tensor2.M != K)
    {
        printf("Error: Dimensiones incompatibles entre tensor1 y tensor2 para maxmin\n");
        return;
    }

    // Asegurar datos en CPU
    TensorResult host_a = tensor1.is_device_ptr ? copy_tensor_to_cpu(tensor1) : tensor1;
    TensorResult host_b = tensor2.is_device_ptr ? copy_tensor_to_cpu(tensor2) : tensor2;
    const float *A = host_a.data;
    const float *B = host_b.data;

    const size_t size_C_min = static_cast<size_t>(batch) * M * N * K;
    const size_t size_C_max = static_cast<size_t>(batch) * M * N;

    float *h_C_min = static_cast<float *>(malloc(size_C_min * sizeof(float)));
    float *h_C_max = static_cast<float *>(malloc(size_C_max * sizeof(float)));

    if (!h_C_min || !h_C_max)
    {
        if (h_C_min)
            free(h_C_min);
        if (h_C_max)
            free(h_C_max);
        printf("Error: No se pudo alocar memoria host para maxmin (CPU)\n");
        return;
    }

    for (int b = 0; b < batch; ++b)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float max_val = -FLT_MAX;
                const size_t base_min = (static_cast<size_t>(b) * M * N + m * N + n) * K;
                for (int k = 0; k < K; ++k)
                {
                    const size_t idx_a = (static_cast<size_t>(b) * M + m) * K + k;
                    const size_t idx_b = (static_cast<size_t>(b) * K + k) * N + n;
                    const float lane_min = std::fmin(A[idx_a], B[idx_b]);
                    h_C_min[base_min + k] = lane_min;
                    if (lane_min > max_val)
                    {
                        max_val = lane_min;
                    }
                }
                h_C_max[static_cast<size_t>(b) * M * N + m * N + n] = max_val;
            }
        }
    }

    max_result = TensorResult(h_C_max, false, batch, M, N, 1, true);
    min_result = TensorResult(h_C_min, false, batch, M, N, K, true);
}

__global__ void maxmin_prima_indices_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    const float *__restrict__ gen_tensor,
    float *__restrict__ C_max,
    float *__restrict__ values,
    float *__restrict__ indices,
    int *output_count,
    const float threshold,
    const int batch,
    const int M,
    const int N,
    const int K)
{
    extern __shared__ unsigned char shared_buffer[];
    float *mins = reinterpret_cast<float *>(shared_buffer);
    int *match_slots = reinterpret_cast<int *>(mins + blockDim.x);

    __shared__ float s_block_max;
    __shared__ float s_prima;
    __shared__ int s_local_count;
    __shared__ int s_global_offset;

    const int k = threadIdx.x;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int b = blockIdx.z;

    if (b >= batch || m >= M || n >= N)
    {
        return;
    }

    const int idx3d = ((b * M) + m) * N + n;
    const int base_a = (b * M + m) * K;
    const int base_b = b * K * N;

    float lane_min = -FLT_MAX;

    if (k < K)
    {
        const float a_val = A[base_a + k];
        const float b_val = B[base_b + k * N + n];
        lane_min = fminf(a_val, b_val);
        mins[k] = lane_min;
    }
    else if (k < blockDim.x)
    {
        mins[k] = -FLT_MAX;
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (k < stride)
        {
            const float other = mins[k + stride];
            if (other > mins[k])
            {
                mins[k] = other;
            }
        }
        __syncthreads();
    }

    if (k == 0)
    {
        s_block_max = mins[0];
        C_max[idx3d] = s_block_max;
        const float gen_val = gen_tensor[idx3d];
        s_prima = s_block_max - gen_val;
        s_local_count = 0;
        s_global_offset = 0;
    }
    __syncthreads();

    if (s_prima <= threshold)
    {
        return;
    }

    int slot = -1;
    constexpr float EPSILON = 1e-6f;

    if (k < K && fabsf(lane_min - s_block_max) <= EPSILON)
    {
        slot = atomicAdd(&s_local_count, 1);
        match_slots[slot] = k;
    }
    __syncthreads();

    if (k == 0 && s_local_count > 0)
    {
        s_global_offset = atomicAdd(output_count, s_local_count);
    }
    __syncthreads();

    if (s_local_count == 0)
    {
        return;
    }

    if (k < s_local_count)
    {
        const int output_pos = s_global_offset + k;
        const int matched_k = match_slots[k];

        indices[output_pos * 4 + 0] = static_cast<float>(b);
        indices[output_pos * 4 + 1] = static_cast<float>(m);
        indices[output_pos * 4 + 2] = static_cast<float>(matched_k);
        indices[output_pos * 4 + 3] = static_cast<float>(n);
        values[output_pos] = s_prima;
    }
}

// Versión mejorada de maxmin que usa TensorResult y retorna tanto max como min
void maxmin(const TensorResult &tensor1, const TensorResult &tensor2,
            TensorResult &max_result, TensorResult &min_result,
            bool keep_in_device)
{
    const bool force_gpu = false; // CPU por defecto; habilitar GPU solo si se reintroduce una bandera explícita
    if (!force_gpu)
    {
        maxmin_cpu(tensor1, tensor2, max_result, min_result);
        return;
    }

    // Validar que los tensores sean 3D (K=1) como espera el kernel
    if (tensor1.K != 1 || tensor2.K != 1)
    {
        printf("Error: maxmin_kernel_v1 solo acepta tensores 3D (K=1)\n");
        return; // tensor nulo
    }

    // Para el kernel, necesitamos que A sea [batch, M, K] y B sea [batch, K, N]
    // Pero como K=1, efectivamente son [batch, M] y [batch, N]
    int batch = tensor1.batch;
    int M = tensor1.M;
    int K = tensor1.N; // En el contexto del kernel, N del tensor1 es K
    int N = tensor2.N;


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
    float *h_A = tensor1.data;
    float *h_B = tensor2.data;
    bool liberar_A = false, liberar_B = false;

    if (tensor1.is_device_ptr)
    {
        h_A = (float *)malloc(size_A);
        cudaMemcpy(h_A, tensor1.data, size_A, cudaMemcpyDeviceToHost);
        liberar_A = true;
    }

    if (tensor2.is_device_ptr)
    {
        h_B = (float *)malloc(size_B);
        cudaMemcpy(h_B, tensor2.data, size_B, cudaMemcpyDeviceToHost);
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
        return ;
    }

    // Crear tensor resultado para C_max (solo retornamos C_max para validación)
    float *h_C_max = (float *)malloc(size_C_max);
    float *h_C_min = (float *)malloc(size_C_min); // Si se quiere retornar min también
    cudaMemcpy(h_C_max, d_C_max, size_C_max, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_min, d_C_min, size_C_min, cudaMemcpyDeviceToHost);

    max_result = TensorResult(h_C_max, false, batch, M, N, 1, true);
    min_result = TensorResult(h_C_min, false, batch, M, N, K, true);
    // Limpiar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_min);
    cudaFree(d_C_max);
    if (liberar_A)
        free(h_A);
    if (liberar_B)
        free(h_B);

    return ;
}

void maxmin_prima_indices(const TensorResult &tensor1, const TensorResult &tensor2,
                          TensorResult &max_result,
                          TensorResult &result_tensor_filtered,
                          TensorResult &result_tensor_values,
                          float threshold,
                          bool keep_in_device)
{
    max_result = TensorResult();
    result_tensor_filtered = TensorResult();
    result_tensor_values = TensorResult();

    if (tensor1.K != 1 || tensor2.K != 1)
    {
        printf("Error: maxmin_prima_indices requiere tensores 3D (K=1)\n");
        return;
    }

    const int batch = tensor1.batch;
    const int M = tensor1.M;
    const int K = tensor1.N;
    const int N = tensor2.N;

    if (tensor2.batch != batch || tensor2.M != K)
    {
        printf("Error: Dimensiones incompatibles entre tensor1 y tensor2 para maxmin_prima_indices\n");
        return;
    }

    if (tensor1.N != tensor2.N)
    {
        printf("Error: maxmin_prima_indices requiere tensores cuadrados (tensor1.N debe coincidir con tensor2.N)\n");
        return;
    }

    const size_t size_A = static_cast<size_t>(batch) * M * K * sizeof(float);
    const size_t size_B = static_cast<size_t>(batch) * K * N * sizeof(float);
    const size_t size_C_max = static_cast<size_t>(batch) * M * N * sizeof(float);
    const int total_elements_3d = batch * M * N;
    const int max_output_size = total_elements_3d * K;

    float *d_A = tensor1.data;
    float *d_B = tensor2.data;
    bool allocated_A = false;
    bool allocated_B = false;

    if (!tensor1.is_device_ptr)
    {
        CHECK_CUDA(cudaMalloc(&d_A, size_A));
        CHECK_CUDA(cudaMemcpy(d_A, tensor1.data, size_A, cudaMemcpyHostToDevice));
        allocated_A = true;
    }

    if (!tensor2.is_device_ptr)
    {
        CHECK_CUDA(cudaMalloc(&d_B, size_B));
        CHECK_CUDA(cudaMemcpy(d_B, tensor2.data, size_B, cudaMemcpyHostToDevice));
        allocated_B = true;
    }

    float *d_C_max = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C_max, size_C_max));

    float *d_values = nullptr;
    float *d_indices = nullptr;
    int *d_output_count = nullptr;

    if (max_output_size > 0)
    {
        CHECK_CUDA(cudaMalloc(&d_values, static_cast<size_t>(max_output_size) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_indices, static_cast<size_t>(max_output_size) * 4 * sizeof(float)));
    }

    CHECK_CUDA(cudaMalloc(&d_output_count, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_output_count, 0, sizeof(int)));

    int block_size = nextPow2(static_cast<unsigned int>(K));
    block_size = std::min(block_size, 1024);
    if (block_size == 0)
    {
        block_size = 1;
    }

    dim3 block(block_size);
    dim3 grid(N, M, batch);
    const size_t shared_mem_size = static_cast<size_t>(block_size) * (sizeof(float) + sizeof(int));

    maxmin_prima_indices_kernel<<<grid, block, shared_mem_size>>>(
        d_A,
        d_B,
        d_A,
        d_C_max,
        d_values,
        d_indices,
        d_output_count,
        threshold,
        batch,
        M,
        N,
        K);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int output_count = 0;
    CHECK_CUDA(cudaMemcpy(&output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (keep_in_device)
    {
        max_result.data = d_C_max;
        max_result.is_device_ptr = true;
        max_result.owns_memory = true;
    }
    else
    {
        float *h_C_max = static_cast<float *>(malloc(size_C_max));
        if (!h_C_max)
        {
            printf("Error: No se pudo alocar memoria host para max_result\n");
        }
        else
        {
            CHECK_CUDA(cudaMemcpy(h_C_max, d_C_max, size_C_max, cudaMemcpyDeviceToHost));
            max_result.data = h_C_max;
            max_result.is_device_ptr = false;
            max_result.owns_memory = true;
        }
        CHECK_CUDA(cudaFree(d_C_max));
    }

    max_result.batch = batch;
    max_result.M = M;
    max_result.N = N;
    max_result.K = 1;

    if (output_count > 0)
    {
        if (keep_in_device)
        {
            float *d_values_final = nullptr;
            float *d_indices_final = nullptr;
            CHECK_CUDA(cudaMalloc(&d_values_final, static_cast<size_t>(output_count) * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_indices_final, static_cast<size_t>(output_count) * 4 * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_values_final, d_values, static_cast<size_t>(output_count) * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_indices_final, d_indices, static_cast<size_t>(output_count) * 4 * sizeof(float), cudaMemcpyDeviceToDevice));

            result_tensor_values.data = d_values_final;
            result_tensor_values.is_device_ptr = true;
            result_tensor_values.owns_memory = true;

            result_tensor_filtered.data = d_indices_final;
            result_tensor_filtered.is_device_ptr = true;
            result_tensor_filtered.owns_memory = true;
        }
        else
        {
            float *h_values = static_cast<float *>(malloc(static_cast<size_t>(output_count) * sizeof(float)));
            float *h_indices = static_cast<float *>(malloc(static_cast<size_t>(output_count) * 4 * sizeof(float)));

            if (!h_values || !h_indices)
            {
                printf("Error: No se pudo alocar memoria host para resultados\n");
                if (h_values)
                    free(h_values);
                if (h_indices)
                    free(h_indices);
            }
            else
            {
                CHECK_CUDA(cudaMemcpy(h_values, d_values, static_cast<size_t>(output_count) * sizeof(float), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_indices, d_indices, static_cast<size_t>(output_count) * 4 * sizeof(float), cudaMemcpyDeviceToHost));

                result_tensor_values.data = h_values;
                result_tensor_values.is_device_ptr = false;
                result_tensor_values.owns_memory = true;

                result_tensor_filtered.data = h_indices;
                result_tensor_filtered.is_device_ptr = false;
                result_tensor_filtered.owns_memory = true;
            }
        }

        result_tensor_values.batch = 1;
        result_tensor_values.M = 1;
        result_tensor_values.N = output_count;
        result_tensor_values.K = 1;

        result_tensor_filtered.batch = 1;
        result_tensor_filtered.M = output_count;
        result_tensor_filtered.N = 4;
        result_tensor_filtered.K = 1;
    }
    else
    {
        printf("No se encontraron elementos que superen el threshold %.4f\n", threshold);
    }

    if (d_values)
        CHECK_CUDA(cudaFree(d_values));
    if (d_indices)
        CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_output_count));

    if (allocated_A)
        CHECK_CUDA(cudaFree(d_A));
    if (allocated_B)
        CHECK_CUDA(cudaFree(d_B));
}
