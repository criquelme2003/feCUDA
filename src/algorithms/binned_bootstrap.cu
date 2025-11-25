#include <vector>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <random>
/*
Kernel 1: Construcción del histograma por columna

Cada columna (o “posición de manzana” en tu tensor [batch, X, Y]) se asigna a un block.

Cada thread del block procesa varios elementos de esa columna.

Incrementa un contador en shared memory usando atomicAdd.

Al final del block, copiamos el histograma parcial al global memory.

Output: histograma [num_bins, columnas]


*/
__global__ void hist_by_x_y(
    const float *__restrict__ data,
    const int num_bins,
    const int M,
    const int N,
    int *__restrict__ histogram)
{
    extern __shared__ int col_histogram[];

    int m = blockIdx.x;
    int n = blockIdx.y;
    int b = threadIdx.x;

    int col_id = (m * N + n) * blockDim.x + b;

    __syncthreads();

    int bin_idx = static_cast<int>(data[b * M * N + m * N + n] * num_bins);

    atomicAdd(&col_histogram[bin_idx], 1);

    __syncthreads();

    if (b == 0)
    {
        for (int bin = 0; bin < num_bins; ++bin)
        {
            histogram[(m * N + n) * num_bins + bin] = col_histogram[bin];
        }
    }
}

__global__ void cdf(
    const int *__restrict__ histogram,
    const int M,
    const int N,
    const int num_bins,
    const int batch_size,
    float *__restrict__ cdf)
{
    extern __shared__ int cum_hist[];

    int m = blockIdx.x;
    int n = blockIdx.y;
    int b = threadIdx.x;

    int col_id = (m * N + n) * num_bins + b;

    cum_hist[b] = histogram[col_id];
    __syncthreads();

    int cdf_val = 0;
    for (int i = 0; i <= b; i++)
    {
        cdf_val += cum_hist[i];
    }

    __syncthreads();

    cdf[(m * N + n) * num_bins + b] = (float)cdf_val / (float)batch_size;
}

/*
Este kernel realiza la interpolación para mapear nuevos valores en base a una lista de probabilidades acumuladas (CDF) y el histograma correspondiente.
Cada thread accede a un percentil y retorna un valor interpolado.

Se instancia 1 bloque por columna, y cada thread maneja una replica, es decir:

- gridDim.x = M (número de filas)
- gridDim.y = N (número de columnas)
- blockDim.x = num_perc (número de percentiles)
- threadIdx.x = índice del percentil actual

Output:

- output: matriz de tamaño [nreps (len),M, N] con los valores interpolados para cada percentil en cada columna.
*/
__global__ void interpolate(
    const float *__restrict__ perc, // [0,1].f
    const float *__restrict__ cdf,
    float *__restrict__ output,
    const int num_bins)
{

    int m = blockIdx.x;
    int n = blockIdx.y;
    int r = threadIdx.x;

    int M = gridDim.x;
    int N = gridDim.y;

    int col_id = (m * gridDim.y + n) * num_bins;

    float p = perc[r];

    int b_idx = num_bins - 1; // default último bin
    for (int i = 0; i < num_bins; i++)
    {
        if (cdf[col_id + i] >= p)
        {
            b_idx = i;
            break;
        }
    }

    int b_1_idx = (b_idx > 0) ? (b_idx - 1) : 0;

    float denom = cdf[col_id + b_idx] - (b_idx > 0 ? cdf[col_id + b_1_idx] : 0.0f);
    float num = p - (b_idx > 0 ? cdf[col_id + b_1_idx] : 0.0f);

    float alpha = (denom > 1e-6f) ? (num / denom) : 1.0f;

    float x = (b_idx + alpha) * (1.0f / num_bins);

    output[r * M * N + m * N + n] = x;
}

void bootstrap(const int num_bins,
               const int M,
               const int N,
               const int batch_size,
               const int rep,
               const float *data,
               float *out_data,
               float *perc)
{
    int *d_histogram;
    float *d_cdf;
    
    float *d_data, *d_out_data, *d_perc;
    

    size_t data_size = M * N * batch_size * sizeof(float);
    size_t out_size = M * N * rep * sizeof(float);
    int hist_len = M * N * num_bins;

    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_out_data, out_size);
    cudaMalloc(&d_perc, rep * sizeof(float));
    cudaMalloc(&d_histogram, hist_len * sizeof(int));
    cudaMalloc(&d_cdf, hist_len * sizeof(float));

    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_perc, perc, rep * sizeof(float), cudaMemcpyHostToDevice);


    cudaMemset(d_histogram, 0, hist_len * sizeof(int));
    cudaMemset(d_cdf, 0, hist_len * sizeof(float));

    dim3 blocks(M, N);
    dim3 threads(batch_size);
    size_t shared_mem_size = num_bins * sizeof(int);

    dim3 blocksCdf(M, N);
    dim3 threadsCdf(num_bins);
    size_t shared_mem_sizeCdf = num_bins * sizeof(int);
    dim3 blocksI(M, N);
    dim3 threadsI(rep);

    hist_by_x_y<<<blocks, threads, shared_mem_size>>>(d_data, num_bins, M, N, d_histogram);
    cdf<<<blocksCdf, threadsCdf, shared_mem_sizeCdf>>>(d_histogram, M, N, num_bins, batch_size, d_cdf);
    interpolate<<<blocksI, threadsI>>>(d_perc, d_cdf, d_out_data, num_bins);
    cudaDeviceSynchronize();

    cudaMemcpy(out_data, d_out_data, M * N * rep * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_histogram);
    cudaFree(d_cdf);
}
