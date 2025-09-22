#include <utils.cuh>
#include <curand_kernel.h>

// Optimización 1: Generar percentiles directamente en GPU
__global__ void init_curand_states(curandState *state, unsigned long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void generate_percents_gpu(curandState *state, float *percents, int replicas)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < replicas)
    {
        percents[idx] = curand_uniform(&state[idx]);
    }
}

/*
Este kernel ordena los datos de cada columna m,n de la matriz de datos.

Cada bloque maneja una columna m, n

    blockIdx.x -> m (fila)
    blockIdx.y -> n (columna)

    gridDim.x = M (número de filas)
    gridDim.y = N (número de columnas)

    blockSize.x = número de hilos por bloque = nextPow2(batch_size)

Parámetros:

    data: matriz de datos original (batch_size, M, N)
    out_data: matriz de datos de salida (batch_size, M, N)
    B: batch_size


*/

__global__ void bitonic_sort(
    const float *__restrict__ data,
    float *__restrict__ out_data,
    const int B)
{
    extern __shared__ float ordered_m_n[];

    int m = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    int M = gridDim.x;
    int N = gridDim.y;
    int block_size = blockDim.x;

    // Cargar datos en memoria compartida
    if (tid < B)
    {
        ordered_m_n[tid] = data[tid * M * N + m * N + n];
    }
    else
    {
        ordered_m_n[tid] = INFINITY;
    }

    __syncthreads();

    // Bitonic Sort
    for (int stage = 1; stage < block_size; stage <<= 1)
    {
        for (int step = stage; step > 0; step >>= 1)
        {
            int partner = tid ^ step;
            bool ascending = (tid & stage) == 0;

            if (partner > tid)
            {
                float my_val = ordered_m_n[tid];
                float partner_val = ordered_m_n[partner];

                bool should_swap = ascending ? (my_val > partner_val) : (my_val < partner_val);

                if (should_swap)
                {
                    ordered_m_n[tid] = partner_val;
                    ordered_m_n[partner] = my_val;
                }
            }

            __syncthreads();
        }
    }

    // Enderezamiento final
    for (int step = block_size >> 1; step > 0; step >>= 1)
    {
        int partner = tid ^ step;

        if (partner > tid)
        {
            float my_val = ordered_m_n[tid];
            float partner_val = ordered_m_n[partner];

            if (my_val > partner_val)
            {
                ordered_m_n[tid] = partner_val;
                ordered_m_n[partner] = my_val;
            }
        }

        __syncthreads();
    }

    // Guardar datos ordenados en la matriz de salida
    if (tid < B)
    {
        out_data[tid * M * N + m * N + n] = ordered_m_n[tid];
    }
}

/*
Este kernel genera muestras bootstrap según los percentiles dados.
Cada bloque maneja una fila m y una columna n, y cada hilo maneja un percentil

    blockIdx.x -> m (fila)
    blockIdx.y -> n (columna)

    gridDim.x = M (número de filas)
    gridDim.y = N (número de columnas)

    blockSize.x = n de percentiles (replicas)
    threadIdx.x -> percentil

    Parámetros:
        ordered_m_n : matriz de datos ordenados (batch_size, M, N)
        out_data : matriz de datos de salida (replicas, M, N)
        B: batch_size

*/
__global__ void interpolate(
    const float *__restrict__ ordered_m_n, // [B*M*N]
    float *__restrict__ out_data,          // [replicas*M*N]
    const float *__restrict__ percents,    // [replicas]
    int B)
{
    int tid = threadIdx.x;
    int m = blockIdx.x;
    int n = blockIdx.y;
    int M = gridDim.x;
    int N = gridDim.y;

    float p = percents[tid];
    float pos = p * (B - 1);
    int lower_idx = (int)pos;
    int upper_idx = min(lower_idx + 1, B - 1);
    float alpha = pos - lower_idx;

    int data_offset = (m * N + n) * B;
    float value = ordered_m_n[lower_idx * M * N + m * N + n] +
                  alpha * (ordered_m_n[upper_idx * M * N + m * N + n] - ordered_m_n[lower_idx * M * N + m * N + n]);

    out_data[tid * M * N + m * N + n] = value;
}

__global__ void interpolate_coalesced(
    const float *__restrict__ ordered_data,
    float *__restrict__ bootstrap_data,
    const float *__restrict__ percents,
    int B, int M, int N, int replicas) {
    
    // Usar índices para mejor coalescing
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N * replicas;
    
    if (global_tid >= total_elements) return;
    
    // Calcular coordenadas
    int replica_idx = global_tid % replicas;
    int mn_idx = global_tid / replicas;
    int m = mn_idx / N;
    int n = mn_idx % N;
    
    float p = percents[replica_idx];
    float pos = p * (B - 1);
    int lower_idx = (int)pos;
    int upper_idx = min(lower_idx + 1, B - 1);
    float alpha = pos - lower_idx;
    
    int base_offset = m * N + n;
    float lower_val = ordered_data[lower_idx * M * N + base_offset];
    float upper_val = ordered_data[upper_idx * M * N + base_offset];
    
    float interpolated_value = lower_val + alpha * (upper_val - lower_val);
    bootstrap_data[global_tid] = interpolated_value;
}


__global__ void interpolate_optimized(
    const float *__restrict__ ordered_m_n,
    float *__restrict__ out_data,
    curandState *rand_states,
    int B, int replicas)
{
    int m = blockIdx.x;
    int n = blockIdx.y;
    int replica_block = blockIdx.z;
    int tid = threadIdx.x;
    int M = gridDim.x;
    int N = gridDim.y;
    
    // Calcular el índice global de replica
    int global_replica_idx = replica_block * blockDim.x + tid;
    
    // Verificar que estamos dentro del rango
    if (global_replica_idx >= replicas)
        return;
    
    // CORRECIÓN: Usar un índice separado para los estados de random
    // que sea único para cada (m,n,replica) combinación
    int rand_state_idx = m * N * replicas + n * replicas + global_replica_idx;
    
    // Generar percentil aleatorio
    float p = curand_uniform(&rand_states[rand_state_idx]);
    
    // CORRECCIÓN: Asegurar que pos esté en el rango [0, B-1]
    float pos = p * (float)(B - 1);
    int lower_idx = (int)floorf(pos);
    int upper_idx = min(lower_idx + 1, B - 1);
    float alpha = pos - (float)lower_idx;
    
    // CORRECCIÓN: Acceso correcto a los datos ordenados
    // Los datos están organizados como [batch][M][N]
    int base_addr = m * N + n;  // Posición (m,n) en la matriz
    
    float lower_val = ordered_m_n[lower_idx * M * N + base_addr];
    float upper_val = ordered_m_n[upper_idx * M * N + base_addr];
    
    // Interpolación lineal
    float value = lower_val + alpha * (upper_val - lower_val);
    
    // Escribir resultado
    out_data[global_replica_idx * M * N + base_addr] = value;
}
float *bootstrap_wrapper(float *data, int M, int N, int batch_size, int replicas)
{
    size_t data_size = M * N * batch_size * sizeof(float);
    float *d_data;
    float *d_ordered_data;
    float *d_bootstrap_data;
    curandState *d_rand_states;
    
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_ordered_data, data_size);
    cudaMalloc(&d_bootstrap_data, M * N * replicas * sizeof(float));
    
    // CORRECCIÓN: Asignar memoria para todos los estados de random necesarios
    cudaMalloc(&d_rand_states, M * N * replicas * sizeof(curandState));
    
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    
    // Ordenamiento bitónico
    dim3 gridDim(M, N);
    size_t shared_mem_size = nextPow2(batch_size) * sizeof(float);
    dim3 blockDim(nextPow2(batch_size));
    
    bitonic_sort<<<gridDim, blockDim, shared_mem_size>>>(d_data, d_ordered_data, batch_size);
    cudaDeviceSynchronize();
    
    // CORRECCIÓN: Inicializar estados de random correctamente
    const int RAND_BLOCK_SIZE = 256;
    int total_rand_states = M * N * replicas;
    dim3 grid_rand((total_rand_states + RAND_BLOCK_SIZE - 1) / RAND_BLOCK_SIZE);
    dim3 block_rand(RAND_BLOCK_SIZE);
    
    init_curand_states<<<grid_rand, block_rand>>>(
        d_rand_states, time(NULL), total_rand_states);
    cudaDeviceSynchronize();
    
    // Interpolación
    const int OPTIMAL_BLOCK_SIZE = 256;
    const int num_replica_blocks = (replicas + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
    
    dim3 grid(M, N, num_replica_blocks);
    dim3 block(OPTIMAL_BLOCK_SIZE);
    
    interpolate_optimized<<<grid, block>>>(
        d_ordered_data, d_bootstrap_data, d_rand_states, batch_size, replicas);
    cudaDeviceSynchronize();
    
    // Limpiar memoria
    cudaFree(d_data);
    cudaFree(d_ordered_data);
    cudaFree(d_rand_states);
    
    return d_bootstrap_data;
}