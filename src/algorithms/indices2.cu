#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <utils.cuh>
#include <types.cuh>

// ============= KERNELS =============

// Primer kernel: Contar cuántos elementos válidos produce cada thread
__global__ void count_valid_elements(float *min_res, float *maxmin_prima, float threshold,
                                    int batch, int M, int N, int K, int *counts_per_thread)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_3d = batch * M * N;
    
    if (idx < total_elements_3d)
    {
        counts_per_thread[idx] = 0;  // Inicializar
        
        float maxmin_value = maxmin_prima[idx];
        
        if (maxmin_value > threshold)
        {
            // Convertir índice a coordenadas
            int b = idx / (M * N);
            int m = (idx % (M * N)) / N;
            int n = (idx % (M * N)) % N;
            int base_idx = b * (M * N * K) + m * (N * K) + n * K;
            
            // Encontrar máximo en la dimensión K
            float max_val = -FLT_MAX;
            for (int k = 0; k < K; k++)
            {
                float current_val = min_res[base_idx + k];
                if (current_val > max_val)
                {
                    max_val = current_val;
                }
            }
            
            // Contar cuántas veces se repite el máximo
            const float EPSILON = 1e-6f;
            int local_count = 0;
            for (int k = 0; k < K; k++)
            {
                if (fabsf(min_res[base_idx + k] - max_val) < EPSILON)
                {
                    local_count++;
                }
            }
            
            counts_per_thread[idx] = local_count;
        }
    }
    else if (idx < total_elements_3d)
    {
        counts_per_thread[idx] = 0;
    }
}

// Segundo kernel: Escribir resultados usando offsets calculados
__global__ void write_results(float *min_res, float *maxmin_prima, float *values, float *indices,
                             float threshold, int batch, int M, int N, int K, 
                             int *counts_per_thread, int *prefix_sums)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_3d = batch * M * N;
    
    if (idx < total_elements_3d && counts_per_thread[idx] > 0)
    {
        // Convertir índice a coordenadas
        int b = idx / (M * N);
        int m = (idx % (M * N)) / N;
        int n = (idx % (M * N)) % N;
        
        float maxmin_value = maxmin_prima[idx];
        int base_idx = b * (M * N * K) + m * (N * K) + n * K;
        
        // Encontrar máximo (repetir cálculo - se podría optimizar guardándolo)
        float max_val = -FLT_MAX;
        for (int k = 0; k < K; k++)
        {
            float current_val = min_res[base_idx + k];
            if (current_val > max_val)
            {
                max_val = current_val;
            }
        }
        
        // Escribir todos los resultados para este thread
        int output_start = prefix_sums[idx];
        int local_output_idx = 0;
        
        const float EPSILON = 1e-6f;
        for (int k = 0; k < K; k++)
        {
            if (fabsf(min_res[base_idx + k] - max_val) < EPSILON)
            {
                int write_pos = output_start + local_output_idx;
                
                // Guardar índices en formato [b, m, k, n]
                indices[write_pos * 4 + 0] = (float)b;
                indices[write_pos * 4 + 1] = (float)m;
                indices[write_pos * 4 + 2] = (float)k;
                indices[write_pos * 4 + 3] = (float)n;
                
                // Guardar el valor máximo encontrado
                values[write_pos] = max_val;
                
                local_output_idx++;
            }
        }
    }
}

// Kernel simple para prefix sum (alternativa si no usas Thrust)
__global__ void simple_prefix_sum(int *input, int *output, int n)
{
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cargar datos a memoria compartida
    if (idx < n)
        sdata[tid] = input[idx];
    else
        sdata[tid] = 0;
    __syncthreads();
    
    // Prefix sum dentro del bloque
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int temp = 0;
        if (tid >= stride)
            temp = sdata[tid - stride];
        __syncthreads();
        
        if (tid >= stride)
            sdata[tid] += temp;
        __syncthreads();
    }
    
    // Escribir resultado (exclusive scan)
    if (idx < n)
    {
        if (tid == 0)
            output[idx] = 0;
        else
            output[idx] = sdata[tid - 1];
    }
}

// ============= FUNCIÓN PRINCIPAL =============

void indices_two_pass(const TensorResult &min_result, const TensorResult &maxmin_prima,
                     TensorResult &result_tensor_filtered, TensorResult &result_tensor_values,
                     float threshold)
{
    // Inicializar resultados como vacíos
    result_tensor_filtered = TensorResult();
    result_tensor_values = TensorResult();

    // Verificar CUDA
    cudaError_t deviceError = cudaDeviceSynchronize();
    if (deviceError != cudaSuccess)
    {
        printf("Error: El dispositivo CUDA no está disponible: %s\n", cudaGetErrorString(deviceError));
        return;
    }

    // Extraer dimensiones
    int batch = min_result.batch;
    int M = min_result.M;
    int N = min_result.N;
    int K = min_result.K;

    // Verificar dimensiones
    if (batch != maxmin_prima.batch || M != maxmin_prima.M || N != maxmin_prima.N)
    {
        printf("Error: Las dimensiones no coinciden\n");
        return;
    }

    int total_elements_3d = batch * M * N;
    int total_elements_4d = batch * M * N * K;

    // Variables para manejo de memoria
    float *d_min_res = nullptr;
    float *d_maxmin_prima = nullptr;
    int *d_counts = nullptr;
    int *d_prefix_sums = nullptr;
    bool allocated_min_res = false;
    bool allocated_maxmin_prima = false;
    bool success = true;

    printf("Procesando tensor con dimensiones: %dx%dx%dx%d\n", batch, M, N, K);
    printf("Total elementos 3D: %d\n", total_elements_3d);

    do {
        // ========== PREPARAR DATOS DE ENTRADA ==========
        
        // Preparar min_result
        if (min_result.is_device_ptr)
        {
            d_min_res = min_result.data;
        }
        else
        {
            if (cudaMalloc(&d_min_res, total_elements_4d * sizeof(float)) != cudaSuccess)
            {
                printf("Error: No se pudo alocar memoria para min_res\n");
                success = false;
                break;
            }
            allocated_min_res = true;
            
            if (cudaMemcpy(d_min_res, min_result.data, total_elements_4d * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            {
                printf("Error: No se pudo copiar min_res a device\n");
                success = false;
                break;
            }
        }

        // Preparar maxmin_prima
        if (maxmin_prima.is_device_ptr)
        {
            d_maxmin_prima = maxmin_prima.data;
        }
        else
        {
            if (cudaMalloc(&d_maxmin_prima, total_elements_3d * sizeof(float)) != cudaSuccess)
            {
                printf("Error: No se pudo alocar memoria para maxmin_prima\n");
                success = false;
                break;
            }
            allocated_maxmin_prima = true;
            
            if (cudaMemcpy(d_maxmin_prima, maxmin_prima.data, total_elements_3d * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            {
                printf("Error: No se pudo copiar maxmin_prima a device\n");
                success = false;
                break;
            }
        }

        // ========== ALOCAR ARRAYS AUXILIARES ==========
        
        if (cudaMalloc(&d_counts, total_elements_3d * sizeof(int)) != cudaSuccess)
        {
            printf("Error: No se pudo alocar memoria para counts\n");
            success = false;
            break;
        }

        if (cudaMalloc(&d_prefix_sums, total_elements_3d * sizeof(int)) != cudaSuccess)
        {
            printf("Error: No se pudo alocar memoria para prefix_sums\n");
            success = false;
            break;
        }

        // ========== PASO 1: CONTAR ELEMENTOS VÁLIDOS ==========
        
        int block_size = 256;
        int grid_size = (total_elements_3d + block_size - 1) / block_size;

        printf("Ejecutando paso 1: conteo (grid=%d, block=%d)\n", grid_size, block_size);

        count_valid_elements<<<grid_size, block_size>>>(
            d_min_res, d_maxmin_prima, threshold, batch, M, N, K, d_counts
        );

        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            printf("Error: Fallo en kernel de conteo\n");
            success = false;
            break;
        }

        // ========== PASO 2: CALCULAR PREFIX SUM ==========
        
        printf("Ejecutando paso 2: prefix sum\n");

        // Opción A: Usar implementación simple (para bloques pequeños)
        if (total_elements_3d <= 1024)
        {
            simple_prefix_sum<<<1, total_elements_3d, total_elements_3d * sizeof(int)>>>(
                d_counts, d_prefix_sums, total_elements_3d
            );
        }
        else
        {
            // Opción B: Implementación básica en CPU (para simplicidad)
            int *h_counts = (int*)malloc(total_elements_3d * sizeof(int));
            int *h_prefix_sums = (int*)malloc(total_elements_3d * sizeof(int));
            
            if (cudaMemcpy(h_counts, d_counts, total_elements_3d * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                printf("Error: No se pudo copiar counts a host\n");
                free(h_counts);
                free(h_prefix_sums);
                success = false;
                break;
            }
            
            // Calcular prefix sum en CPU
            h_prefix_sums[0] = 0;
            for (int i = 1; i < total_elements_3d; i++)
            {
                h_prefix_sums[i] = h_prefix_sums[i-1] + h_counts[i-1];
            }
            
            if (cudaMemcpy(d_prefix_sums, h_prefix_sums, total_elements_3d * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
            {
                printf("Error: No se pudo copiar prefix_sums a device\n");
                free(h_counts);
                free(h_prefix_sums);
                success = false;
                break;
            }
            
            // Obtener total de outputs
            int total_outputs = h_prefix_sums[total_elements_3d-1] + h_counts[total_elements_3d-1];
            printf("Total de outputs esperados: %d\n", total_outputs);
            
            free(h_counts);
            free(h_prefix_sums);
            
            if (total_outputs == 0)
            {
                printf("No hay elementos que superen el threshold %.4f\n", threshold);
                success = true;  // No es error, simplemente no hay resultados
                break;
            }

            // ========== PASO 3: ALOCAR MEMORIA PARA RESULTADOS ==========
            
            float *d_values, *d_indices;
            if (cudaMalloc(&d_values, total_outputs * sizeof(float)) != cudaSuccess)
            {
                printf("Error: No se pudo alocar memoria para values\n");
                success = false;
                break;
            }

            if (cudaMalloc(&d_indices, total_outputs * 4 * sizeof(float)) != cudaSuccess)
            {
                printf("Error: No se pudo alocar memoria para indices\n");
                cudaFree(d_values);
                success = false;
                break;
            }

            // ========== PASO 4: ESCRIBIR RESULTADOS ==========
            
            printf("Ejecutando paso 3: escritura de resultados\n");

            write_results<<<grid_size, block_size>>>(
                d_min_res, d_maxmin_prima, d_values, d_indices,
                threshold, batch, M, N, K, d_counts, d_prefix_sums
            );

            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                printf("Error: Fallo en kernel de escritura\n");
                cudaFree(d_values);
                cudaFree(d_indices);
                success = false;
                break;
            }

            // ========== PASO 5: COPIAR RESULTADOS A HOST ==========
            
            float *h_values = (float*)malloc(total_outputs * sizeof(float));
            float *h_indices = (float*)malloc(total_outputs * 4 * sizeof(float));

            if (!h_values || !h_indices)
            {
                printf("Error: No se pudo alocar memoria host\n");
                if (h_values) free(h_values);
                if (h_indices) free(h_indices);
                cudaFree(d_values);
                cudaFree(d_indices);
                success = false;
                break;
            }

            if (cudaMemcpy(h_values, d_values, total_outputs * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(h_indices, d_indices, total_outputs * 4 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                printf("Error: No se pudieron copiar resultados a host\n");
                free(h_values);
                free(h_indices);
                cudaFree(d_values);
                cudaFree(d_indices);
                success = false;
                break;
            }

            // Limpiar memoria device de resultados
            cudaFree(d_values);
            cudaFree(d_indices);

            // ========== CONFIGURAR TENSORRESULT DE SALIDA ==========
            
            result_tensor_values.data = h_values;
            result_tensor_values.is_device_ptr = false;
            result_tensor_values.owns_memory = true;
            result_tensor_values.batch = 1;
            result_tensor_values.M = 1;
            result_tensor_values.N = total_outputs;
            result_tensor_values.K = 1;

            result_tensor_filtered.data = h_indices;
            result_tensor_filtered.is_device_ptr = false;
            result_tensor_filtered.owns_memory = true;
            result_tensor_filtered.batch = 1;
            result_tensor_filtered.M = total_outputs;
            result_tensor_filtered.N = 4;
            result_tensor_filtered.K = 1;

            printf("Procesamiento completado exitosamente. Outputs: %d\n", total_outputs);
        }

    } while (false);

    // ========== LIMPIAR MEMORIA ==========
    
    if (allocated_min_res && d_min_res)
        cudaFree(d_min_res);
    if (allocated_maxmin_prima && d_maxmin_prima)
        cudaFree(d_maxmin_prima);
    if (d_counts)
        cudaFree(d_counts);
    if (d_prefix_sums)
        cudaFree(d_prefix_sums);

    if (!success)
    {
        printf("Error: El procesamiento falló\n");
        // Limpiar cualquier memoria host que se haya alocado
        if (result_tensor_values.owns_memory && result_tensor_values.data)
        {
            free(result_tensor_values.data);
            result_tensor_values = TensorResult();
        }
        if (result_tensor_filtered.owns_memory && result_tensor_filtered.data)
        {
            free(result_tensor_filtered.data);
            result_tensor_filtered = TensorResult();
        }
    }
}