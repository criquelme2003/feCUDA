#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <cstring>
#include <utils.cuh>
#include <types.cuh>
#include <algorithm>

__global__ void strainer(float *min_res,
                         float *maxmin_prima,
                         float *values,
                         float *indices,
                         float threshold,
                         int batch,
                         int M, int N, int K,
                         int *output_count,
                         int n_offset)
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
                    indices[output_pos * 4 + 3] = (float)(n + n_offset); // N global

                    // Guardar el valor máximo en values
                    values[output_pos] = maxmin_value;
                }
            }
        }
    }
}

namespace
{
    // Estima chunk_n máximo dado el free_mem disponible. Devuelve al menos 1.
    int compute_chunk_n_for_batch1(int M, int N, int K, size_t free_mem_bytes)
    {
        // Para un chunk de tamaño chunk_n, los buffers principales son:
        // min_chunk (M * chunk_n * K), values (M * chunk_n * K),
        // indices (4 * M * chunk_n * K) -> total aprox 6 * M * chunk_n * K floats.
        const double bytes_per_unit_n = 6.0 * static_cast<double>(M) * static_cast<double>(K) * sizeof(float);
        if (bytes_per_unit_n <= 0.0)
        {
            return 1;
        }
        const double max_chunk_n = static_cast<double>(free_mem_bytes) * 0.6 / bytes_per_unit_n;
        int chunk_n = static_cast<int>(max_chunk_n);
        if (chunk_n < 1)
            chunk_n = 1;
        chunk_n = std::min(chunk_n, N);
        return chunk_n;
    }

    // Procesa en columnas (chunk_n) cuando batch=1 para reducir consumo de memoria.
    bool indices_chunked_batch1(const TensorResult &min_result, const TensorResult &maxmin_prima,
                                TensorResult &result_tensor_filtered, TensorResult &result_tensor_values,
                                float threshold)
    {
        const int batch = min_result.batch;
        const int M = min_result.M;
        const int N = min_result.N;
        const int K = min_result.K;

        if (batch != 1 || M <= 0 || N <= 0 || K <= 0)
        {
            return false;
        }

        size_t free_mem = 0;
        size_t total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess || free_mem == 0)
        {
            free_mem = 256 * 1024 * 1024; // fallback conservador
        }
        int chunk_n = compute_chunk_n_for_batch1(M, N, K, free_mem);
        if (chunk_n <= 0)
        {
            return false;
        }

        std::vector<float> h_values;
        std::vector<float> h_indices;
        h_values.reserve(static_cast<size_t>(M) * N); // estimación mínima
        h_indices.reserve(static_cast<size_t>(M) * N * 4);

        // Buffers temporales host para copiar slices cuando vienen de host
        const bool min_on_device = min_result.is_device_ptr;
        const bool max_on_device = maxmin_prima.is_device_ptr;

        for (int n_start = 0; n_start < N; n_start += chunk_n)
        {
            const int current_chunk_n = std::min(chunk_n, N - n_start);
            const int chunk_elems_min = M * current_chunk_n * K;
            const int chunk_elems_max = M * current_chunk_n;
            const int chunk_elems_3d = M * current_chunk_n;
            const size_t bytes_min = static_cast<size_t>(chunk_elems_min) * sizeof(float);
            const size_t bytes_max = static_cast<size_t>(chunk_elems_max) * sizeof(float);

            float *d_min_chunk = nullptr;
            float *d_max_chunk = nullptr;
            float *d_values = nullptr;
            float *d_indices = nullptr;
            int *d_output_count = nullptr;

            if (cudaMalloc(&d_min_chunk, bytes_min) != cudaSuccess ||
                cudaMalloc(&d_max_chunk, bytes_max) != cudaSuccess ||
                cudaMalloc(&d_values, bytes_min) != cudaSuccess ||                           // mismo tamaño que min
                cudaMalloc(&d_indices, static_cast<size_t>(chunk_elems_min) * 4 * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d_output_count, sizeof(int)) != cudaSuccess)
            {
                cudaFree(d_min_chunk);
                cudaFree(d_max_chunk);
                cudaFree(d_values);
                cudaFree(d_indices);
                cudaFree(d_output_count);
                return false;
            }

            cudaMemset(d_output_count, 0, sizeof(int));

            // Copiar slice de maxmin_prima (shape [M, N]) para columnas [n_start, n_start+current_chunk_n)
            if (max_on_device)
            {
                for (int m = 0; m < M; ++m)
                {
                    const size_t offset = static_cast<size_t>(m) * N + n_start;
                    const float *src = maxmin_prima.data + offset;
                    float *dst = d_max_chunk + static_cast<size_t>(m) * current_chunk_n;
                    cudaMemcpy(dst, src, static_cast<size_t>(current_chunk_n) * sizeof(float), cudaMemcpyDeviceToDevice);
                }
            }
            else
            {
                for (int m = 0; m < M; ++m)
                {
                    const size_t offset = static_cast<size_t>(m) * N + n_start;
                    const float *src = maxmin_prima.data + offset;
                    float *dst = d_max_chunk + static_cast<size_t>(m) * current_chunk_n;
                    cudaMemcpy(dst, src, static_cast<size_t>(current_chunk_n) * sizeof(float), cudaMemcpyHostToDevice);
                }
            }

            // Copiar slice de min_result (shape [M, N, K]) para columnas del chunk
            if (min_on_device)
            {
                for (int m = 0; m < M; ++m)
                {
                    const size_t offset = static_cast<size_t>(m) * N * K + static_cast<size_t>(n_start) * K;
                    const float *src = min_result.data + offset;
                    float *dst = d_min_chunk + static_cast<size_t>(m) * current_chunk_n * K;
                    cudaMemcpy(dst, src, static_cast<size_t>(current_chunk_n) * K * sizeof(float), cudaMemcpyDeviceToDevice);
                }
            }
            else
            {
                for (int m = 0; m < M; ++m)
                {
                    const size_t offset = static_cast<size_t>(m) * N * K + static_cast<size_t>(n_start) * K;
                    const float *src = min_result.data + offset;
                    float *dst = d_min_chunk + static_cast<size_t>(m) * current_chunk_n * K;
                    cudaMemcpy(dst, src, static_cast<size_t>(current_chunk_n) * K * sizeof(float), cudaMemcpyHostToDevice);
                }
            }

            int block_size = 256;
            int grid_size = (chunk_elems_3d + block_size - 1) / block_size;
            const size_t shared_mem_size = static_cast<size_t>(block_size) * (sizeof(float) + sizeof(int));

            strainer<<<grid_size, block_size, shared_mem_size>>>(
                d_min_chunk,
                d_max_chunk,
                d_values,
                d_indices,
                threshold,
                1,
                M,
                current_chunk_n,
                K,
                d_output_count,
                n_start);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            int output_count = 0;
            CHECK_CUDA(cudaMemcpy(&output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost));
            if (output_count > 0)
            {
                const size_t values_bytes = static_cast<size_t>(output_count) * sizeof(float);
                const size_t indices_bytes = static_cast<size_t>(output_count) * 4 * sizeof(float);
                const size_t values_offset = h_values.size();
                const size_t indices_offset = h_indices.size();
                h_values.resize(values_offset + output_count);
                h_indices.resize(indices_offset + static_cast<size_t>(output_count) * 4);
                CHECK_CUDA(cudaMemcpy(h_values.data() + values_offset, d_values, values_bytes, cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_indices.data() + indices_offset, d_indices, indices_bytes, cudaMemcpyDeviceToHost));
            }

            cudaFree(d_min_chunk);
            cudaFree(d_max_chunk);
            cudaFree(d_values);
            cudaFree(d_indices);
            cudaFree(d_output_count);
        }

        if (h_values.empty())
        {
            result_tensor_filtered = TensorResult();
            result_tensor_values = TensorResult();
            return true;
        }

        // Construir TensorResult en CPU
        float *indices_buf = static_cast<float *>(malloc(h_indices.size() * sizeof(float)));
        float *values_buf = static_cast<float *>(malloc(h_values.size() * sizeof(float)));
        if (!indices_buf || !values_buf)
        {
            if (indices_buf)
                free(indices_buf);
            if (values_buf)
                free(values_buf);
            return false;
        }
        std::memcpy(indices_buf, h_indices.data(), h_indices.size() * sizeof(float));
        std::memcpy(values_buf, h_values.data(), h_values.size() * sizeof(float));

        result_tensor_filtered.data = indices_buf;
        result_tensor_filtered.is_device_ptr = false;
        result_tensor_filtered.owns_memory = true;
        result_tensor_filtered.batch = 1;
        result_tensor_filtered.M = static_cast<int>(h_values.size());
        result_tensor_filtered.N = 4;
        result_tensor_filtered.K = 1;

        result_tensor_values.data = values_buf;
        result_tensor_values.is_device_ptr = false;
        result_tensor_values.owns_memory = true;
        result_tensor_values.batch = 1;
        result_tensor_values.M = 1;
        result_tensor_values.N = static_cast<int>(h_values.size());
        result_tensor_values.K = 1;
        return true;
    }
} // namespace

void indices(const TensorResult &min_result, const TensorResult &maxmin_prima,
             TensorResult &result_tensor_filtered, TensorResult &result_tensor_values,
             float threshold, bool keep_in_device){
    // Inicializar resultados como vacíos
    result_tensor_filtered = TensorResult();
    result_tensor_values = TensorResult();

    // Verificar estado previo del dispositivo CUDA
    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess)
    {
        printf("Error previo en indices.cu (estado CUDA): %s\n", cudaGetErrorString(sync_status));
        return;
    }

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

    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t meminfo_status = cudaMemGetInfo(&free_memory, &total_memory);
    if (meminfo_status != cudaSuccess)
    {
        printf("Advertencia: cudaMemGetInfo falló en indices.cu (%s)\n", cudaGetErrorString(meminfo_status));
        free_memory = total_memory = 0;
    }

    // Ruta optimizada para batch=1 con chunking por columnas para reducir memoria
    if (batch == 1)
    {
        if (indices_chunked_batch1(min_result, maxmin_prima, result_tensor_filtered, result_tensor_values, threshold))
        {
            return;
        }
    }

    auto bytes_needed_for = [&](int batch_count) -> size_t
    {
        size_t inputs = static_cast<size_t>(batch_count) * M * N * (sizeof(float) * (K + 1));
        size_t outputs = static_cast<size_t>(batch_count) * M * N * K * (sizeof(float) * 5);
        return inputs + outputs + sizeof(int);
    };

    int chunk_batch = batch;
    if (free_memory > 0)
    {
        for (int btest = 1; btest <= batch; ++btest)
        {
            if (bytes_needed_for(btest) > static_cast<size_t>(free_memory * 0.6))
            {
                chunk_batch = std::max(1, btest - 1);
                break;
            }
        }
    }

    std::vector<float> h_values_accum;
    std::vector<float> h_indices_accum;
    int total_output_count = 0;
    int processed = 0;

    while (processed < batch)
    {
        int current_batch = std::min(chunk_batch, batch - processed);
        bool chunk_done = false;

        while (current_batch > 0 && !chunk_done)
        {
            int chunk_elems_3d = current_batch * M * N;
            int chunk_elems_4d = chunk_elems_3d * K;
            int chunk_max_output = chunk_elems_3d * K;

            float *d_values = nullptr;
            float *d_indices = nullptr;
            int *d_output_count = nullptr;
            cudaError_t alloc_val = cudaMalloc(&d_values, static_cast<size_t>(chunk_max_output) * sizeof(float));
            cudaError_t alloc_idx = (alloc_val == cudaSuccess) ? cudaMalloc(&d_indices, static_cast<size_t>(chunk_max_output) * 4 * sizeof(float)) : alloc_val;
            cudaError_t alloc_cnt = (alloc_idx == cudaSuccess) ? cudaMalloc(&d_output_count, sizeof(int)) : alloc_idx;
            if (alloc_cnt != cudaSuccess)
            {
                printf("Error: memoria insuficiente en chunk indices (batch=%d, offset=%d): %s\n", current_batch, processed, cudaGetErrorString(alloc_cnt));
                if (alloc_val == cudaSuccess) cudaFree(d_values);
                if (alloc_idx == cudaSuccess) cudaFree(d_indices);
                current_batch /= 2;
                continue;
            }
            cudaMemset(d_output_count, 0, sizeof(int));

            const float *min_ptr = min_result.data + static_cast<size_t>(processed) * M * N * K;
            const float *max_ptr = maxmin_prima.data + static_cast<size_t>(processed) * M * N;

            float *d_min_res = nullptr;
            float *d_maxmin_prima = nullptr;
            bool free_min = false, free_max = false;

            if (min_result.is_device_ptr)
            {
                d_min_res = const_cast<float *>(min_ptr);
            }
            else
            {
                cudaError_t a = cudaMalloc(&d_min_res, static_cast<size_t>(chunk_elems_4d) * sizeof(float));
                if (a != cudaSuccess)
                {
                    printf("Error: memoria insuficiente min_res chunk (batch=%d, offset=%d): %s\n", current_batch, processed, cudaGetErrorString(a));
                    if (alloc_val == cudaSuccess) cudaFree(d_values);
                    if (alloc_idx == cudaSuccess) cudaFree(d_indices);
                    cudaFree(d_output_count);
                    current_batch /= 2;
                    continue;
                }
                cudaMemcpy(d_min_res, min_ptr, static_cast<size_t>(chunk_elems_4d) * sizeof(float), cudaMemcpyHostToDevice);
                free_min = true;
            }

            if (maxmin_prima.is_device_ptr)
            {
                d_maxmin_prima = const_cast<float *>(max_ptr);
            }
            else
            {
                cudaError_t a = cudaMalloc(&d_maxmin_prima, static_cast<size_t>(chunk_elems_3d) * sizeof(float));
                if (a != cudaSuccess)
                {
                    printf("Error: memoria insuficiente maxmin_prima chunk (batch=%d, offset=%d): %s\n", current_batch, processed, cudaGetErrorString(a));
                    if (free_min) cudaFree(d_min_res);
                    if (alloc_val == cudaSuccess) cudaFree(d_values);
                    if (alloc_idx == cudaSuccess) cudaFree(d_indices);
                    cudaFree(d_output_count);
                    current_batch /= 2;
                    continue;
                }
                cudaMemcpy(d_maxmin_prima, max_ptr, static_cast<size_t>(chunk_elems_3d) * sizeof(float), cudaMemcpyHostToDevice);
                free_max = true;
            }

            int block_size = 256;
            int grid_size = (chunk_elems_3d + block_size - 1) / block_size;
            strainer<<<grid_size, block_size>>>(d_min_res, d_maxmin_prima, d_values, d_indices, threshold, current_batch, M, N, K, d_output_count, 0);
            cudaError_t sync = cudaDeviceSynchronize();
            if (sync != cudaSuccess)
            {
                printf("Error durante kernel chunk en indices.cu (batch=%d, offset=%d): %s\n", current_batch, processed, cudaGetErrorString(sync));
                if (free_min) cudaFree(d_min_res);
                if (free_max) cudaFree(d_maxmin_prima);
                cudaFree(d_values); cudaFree(d_indices); cudaFree(d_output_count);
                current_batch /= 2;
                continue;
            }

            int output_count = 0;
            cudaMemcpy(&output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (output_count > chunk_max_output)
            {
                printf("Error: output_count excede capacidad del chunk (batch=%d, offset=%d)\n", current_batch, processed);
                if (free_min) cudaFree(d_min_res);
                if (free_max) cudaFree(d_maxmin_prima);
                cudaFree(d_values); cudaFree(d_indices); cudaFree(d_output_count);
                current_batch /= 2;
                continue;
            }

            if (output_count > 0)
            {
                size_t old = h_values_accum.size();
                h_values_accum.resize(old + output_count);
                h_indices_accum.resize((old + output_count) * 4);
                cudaMemcpy(h_values_accum.data() + old, d_values, static_cast<size_t>(output_count) * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_indices_accum.data() + old * 4, d_indices, static_cast<size_t>(output_count) * 4 * sizeof(float), cudaMemcpyDeviceToHost);
                for (size_t i = old; i < static_cast<size_t>(old + output_count); ++i)
                {
                    h_indices_accum[i * 4 + 0] += static_cast<float>(processed);
                }
                total_output_count += output_count;
            }

            if (free_min) cudaFree(d_min_res);
            if (free_max) cudaFree(d_maxmin_prima);
            cudaFree(d_values); cudaFree(d_indices); cudaFree(d_output_count);

            processed += current_batch;
            chunk_done = true;
        }

        if (!chunk_done)
        {
            printf("Error: no se pudo procesar chunk en indices (offset=%d)\n", processed);
            break;
        }
    }

    if (total_output_count == 0)
    {
        return;
    }

    if (keep_in_device)
    {
        float *d_values_final = nullptr;
        float *d_indices_final = nullptr;
        CHECK_CUDA(cudaMalloc(&d_values_final, static_cast<size_t>(total_output_count) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_indices_final, static_cast<size_t>(total_output_count) * 4 * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_values_final, h_values_accum.data(), static_cast<size_t>(total_output_count) * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_indices_final, h_indices_accum.data(), static_cast<size_t>(total_output_count) * 4 * sizeof(float), cudaMemcpyHostToDevice));

        result_tensor_filtered = TensorResult(d_indices_final, true, 1, total_output_count, 4, 1, true);
        result_tensor_values = TensorResult(d_values_final, true, 1, 1, total_output_count, 1, true);
    }
    else
    {
        float *h_values = (float *)malloc(static_cast<size_t>(total_output_count) * sizeof(float));
        float *h_indices = (float *)malloc(static_cast<size_t>(total_output_count) * 4 * sizeof(float));
        if (!h_values || !h_indices)
        {
            if (h_values) free(h_values);
            if (h_indices) free(h_indices);
            printf("Error: No se pudo alocar memoria host para resultados\n");
            return;
        }
        std::memcpy(h_values, h_values_accum.data(), static_cast<size_t>(total_output_count) * sizeof(float));
        std::memcpy(h_indices, h_indices_accum.data(), static_cast<size_t>(total_output_count) * 4 * sizeof(float));
        result_tensor_filtered = TensorResult(h_indices, false, 1, total_output_count, 4, 1, true);
        result_tensor_values = TensorResult(h_values, false, 1, 1, total_output_count, 1, true);
    }
}
