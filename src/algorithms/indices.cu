#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cmath>
#include <utils.cuh>
#include <core/types.cuh>

__global__ void strainer(float *min_res,
                         float *maxmin_prima,
                         float *values,
                         float *indices,
                         float threshold,
                         int batch,
                         int M, int N, int K,
                         int *output_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_3d = batch * M * N; // Total elementos en maxmin_prima (3D)

    if (idx < total_elements_3d)
    {
        // Convertir índice lineal a coordenadas (b, m, n) para maxmin_prima
        int b = idx / (M * N);
        int m = (idx % (M * N)) / N;
        int n = (idx % (M * N)) % N;

        // Obtener valor de maxmin_prima en posición (b, m, n)
        float maxmin_value = maxmin_prima[idx];

        // Verificar si supera el threshold
        if (maxmin_value > threshold)
        {
            // Buscar en min_res en la misma posición (b, m, n) pero en todas las K
            // min_res tiene dimensiones [batch, M, N, K]

            int base_idx = b * (M * N * K) + m * (N * K) + n * K;
            // Buscar maximo en min_res
            float max_val = -FLT_MAX;
            for (int k = 0; k < K; k++)
            {
                float current_val = min_res[base_idx + k];
                if (current_val > max_val)
                {
                    max_val = current_val;
                }
            }

            // Inicializar base_idx para acceder a min_res
            // Contar cuántas veces se repite este máximo y guardar índices
            for (int k = 0; k < K; k++)
            {
                float current_val = min_res[base_idx + k];
                const float EPSILON = 1e-6f;
                if (fabsf(current_val - max_val) < EPSILON)
                {
                    // Obtener posición en el array de salida usando atomic add
                    int output_pos = atomicAdd(output_count, 1);

                    // Guardar índices 4D [b, m, k, n] en el array indices
                    // Cada elemento ocupa 4 posiciones en el array indices
                    indices[output_pos * 4 + 0] = (float)b; // batch
                    indices[output_pos * 4 + 1] = (float)m; // M
                    indices[output_pos * 4 + 2] = (float)k; // K (donde está el máximo)
                    indices[output_pos * 4 + 3] = (float)n; // N

                    // Guardar el valor máximo en values
                    values[output_pos] = maxmin_value;
                }
            }
        }
    }
}

void indices(const TensorResult &min_result, const TensorResult &maxmin_prima,
             TensorResult &result_tensor_filtered, TensorResult &result_tensor_values,
             float threshold, bool keep_in_device)
{
    // Inicializar resultados como vacíos
    result_tensor_filtered = TensorResult();
    result_tensor_values = TensorResult();

    // Verificar estado del dispositivo CUDA
    CHECK_CUDA(cudaDeviceSynchronize());

    // Extraer dimensiones
    int batch = min_result.batch;
    int M = min_result.M;
    int N = min_result.N;
    int K = min_result.K;

    // Verificar que las dimensiones coincidan (excepto K)
    if (batch != maxmin_prima.batch || M != maxmin_prima.M || N != maxmin_prima.N)
    {
        printf("Error: Las dimensiones no coinciden entre min_result y maxmin_prima\n");
        return;
    }

    // Calcular tamaños
    int total_elements_3d = batch * M * N;
    int total_elements_4d = batch * M * N * K;
    int max_output_size = total_elements_3d * K;

    // Declarar variables
    float *d_min_res = nullptr;
    float *d_maxmin_prima = nullptr;
    float *d_values = nullptr;
    float *d_indices = nullptr;
    int *d_output_count = nullptr;
    bool allocated_min_res = false;
    bool allocated_maxmin_prima = false;

    // Alocar memoria device
    CHECK_CUDA(cudaMalloc(&d_values, max_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, max_output_size * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_count, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_output_count, 0, sizeof(int)));

    // Preparar datos de entrada
    if (min_result.is_device_ptr)
    {
        d_min_res = min_result.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_min_res, total_elements_4d * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_min_res, min_result.data, total_elements_4d * sizeof(float), cudaMemcpyHostToDevice));
        allocated_min_res = true;
    }

    if (maxmin_prima.is_device_ptr)
    {
        d_maxmin_prima = maxmin_prima.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_maxmin_prima, total_elements_3d * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_maxmin_prima, maxmin_prima.data, total_elements_3d * sizeof(float), cudaMemcpyHostToDevice));
        allocated_maxmin_prima = true;
    }

    // Configurar y lanzar kernel
    int block_size = 256;
    int grid_size = (total_elements_3d + block_size - 1) / block_size;
    strainer<<<grid_size, block_size>>>(d_min_res, d_maxmin_prima, d_values, d_indices,threshold, batch, M, N, K, d_output_count);
    
    CHECK_CUDA(cudaDeviceSynchronize());

    // Obtener número de elementos de salida
    int output_count;
    CHECK_CUDA(cudaMemcpy(&output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Procesar resultados
    if (output_count > 0)
    {
        if (keep_in_device)
        {
            // Mantener resultados en GPU con tamaño optimizado
            float *d_values_final, *d_indices_final;
            CHECK_CUDA(cudaMalloc(&d_values_final, output_count * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_indices_final, output_count * 4 * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_values_final, d_values, output_count * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_indices_final, d_indices, output_count * 4 * sizeof(float), cudaMemcpyDeviceToDevice));

            // Configurar TensorResult en GPU
            result_tensor_filtered.data = d_indices_final;
            result_tensor_filtered.is_device_ptr = true;
            result_tensor_filtered.owns_memory = true;
            result_tensor_filtered.batch = 1;
            result_tensor_filtered.M = output_count;
            result_tensor_filtered.N = 4;
            result_tensor_filtered.K = 1;

            result_tensor_values.data = d_values_final;
            result_tensor_values.is_device_ptr = true;
            result_tensor_values.owns_memory = true;
            result_tensor_values.batch = 1;
            result_tensor_values.M = 1;
            result_tensor_values.N = output_count;
            result_tensor_values.K = 1;
        }
        else
        {
            // Transferir resultados a CPU
            float *h_values = (float *)malloc(output_count * sizeof(float));
            float *h_indices = (float *)malloc(output_count * 4 * sizeof(float));

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
                CHECK_CUDA(cudaMemcpy(h_values, d_values, output_count * sizeof(float), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_indices, d_indices, output_count * 4 * sizeof(float), cudaMemcpyDeviceToHost));

                // Configurar TensorResult en CPU
                result_tensor_filtered.data = h_indices;
                result_tensor_filtered.is_device_ptr = false;
                result_tensor_filtered.owns_memory = true;
                result_tensor_filtered.batch = 1;
                result_tensor_filtered.M = output_count;
                result_tensor_filtered.N = 4;
                result_tensor_filtered.K = 1;

                result_tensor_values.data = h_values;
                result_tensor_values.is_device_ptr = false;
                result_tensor_values.owns_memory = true;
                result_tensor_values.batch = 1;
                result_tensor_values.M = 1;
                result_tensor_values.N = output_count;
                result_tensor_values.K = 1;
            }
        }
    }
    else
    {
        printf("No se encontraron elementos que superen el threshold %.4f\n", threshold);
    }

    // Limpiar memoria device
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_output_count));
    if (allocated_min_res)
        CHECK_CUDA(cudaFree(d_min_res));
    if (allocated_maxmin_prima)
        CHECK_CUDA(cudaFree(d_maxmin_prima));
}
