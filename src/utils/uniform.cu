#include <cstdio>
#include <curand_kernel.h>
#include <utils.cuh>
// Elige tamaño múltiplo de 4: el kernel escribe float4
constexpr int TPB = 256;

__global__ void init_states(curandStatePhilox4_32_10_t *states,
                            unsigned long long seed)
{
    const unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    // sequence = tid garantiza subcadenas independientes; offset = 0
    curand_init(seed, /*sequence*/ (unsigned long long)tid, /*offset*/ 0ULL, &states[tid]);
}

__global__ void rng_uniform_float4(curandStatePhilox4_32_10_t *states,
                                   float *__restrict__ out,
                                   size_t n_float)
{
    const unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    // Trabajamos en grupos de 4 floats
    const unsigned long long n_vec4 = n_float / 4ULL;
    for (unsigned long long i = tid; i < n_vec4; i += stride)
    {
        curandStatePhilox4_32_10_t local = states[tid]; // cache en registro
        // Genera 4 uniformes en (0,1]
        float4 u = curand_uniform4(&local);
        // Escribe coalescentemente como float4
        reinterpret_cast<float4 *>(out)[i] = u;
        states[tid] = local; // persiste el estado
    }
}

float *uniform(int n)
{

    // Configura grilla
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    const int blocks = (prop.multiProcessorCount * 8); // heurístico: 8 bloques por SM

    // Memoria
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    // Estados Philox: uno por hilo
    curandStatePhilox4_32_10_t *d_states = nullptr;
    CHECK_CUDA(cudaMalloc(&d_states, blocks * TPB * sizeof(curandStatePhilox4_32_10_t)));

    // Init (hacerlo UNA vez)
    init_states<<<blocks, TPB>>>(d_states, /*seed*/ 123456789ULL);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Uniformes
    rng_uniform_float4<<<blocks, TPB>>>(d_states, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Limpieza
    CHECK_CUDA(cudaFree(d_states));
    return d_out;
}
