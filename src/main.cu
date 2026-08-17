#include "utils.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#include "core/types.cuh"
#include "headers.cuh"

#define RAND_SEED 1111ULL
#define MIN_DIFF  0.01f

// ─── Kernels de utilidad ────────────────────────────────────────────────────

__global__ void FloatToHalfKernel(float *d_in, __half *d_out, int total_elements)
{
    int total_threads = gridDim.x * blockDim.x;
    int globalId      = blockIdx.x * blockDim.x + threadIdx.x; // fix: era gridDim.x
    for (int id = globalId; id < total_elements; id += total_threads)
        d_out[id] = __float2half(d_in[id]);
}

__global__ void randomInit(curandState_t *states, unsigned long long seed, int total_elements)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < total_elements)
        curand_init(seed, globalId, 0, states + globalId);
}

__global__ void randomFill(__half *d_out, int total_elements, curandState_t *states)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < total_elements)
    {
        curandState *state = states + globalId;
        d_out[globalId]    = __float2half(curand_uniform(state));
    }
}

// ─── Referencia CPU ─────────────────────────────────────────────────────────
// Implementación exacta del algoritmo MaxMin en CPU para validación.
// Para matrices cuadradas [B, M, M] con K = M (mismo tensor para A y B).
// Computa: max_k min(A[b,m,k], A[b,k,n]) y cuenta caminos que superan thr.

int cpu_maxmin_count(const std::vector<float> &A, int B, int M, float thr)
{
    int count = 0;
    // A tiene layout [B, M, M] → índice: b*M*M + row*M + col
    for (int b = 0; b < B; b++)
    {
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < M; n++)
            {
                // Calcular max_k min(A[b,m,k], A[b,k,n])
                float k_max = -1e38f;
                for (int k = 0; k < M; k++)
                {
                    float a = A[b * M * M + m * M + k];
                    float bv = A[b * M * M + k * M + n];
                    float mn = (a < bv) ? a : bv;
                    if (mn > k_max) k_max = mn;
                }

                float orig = A[b * M * M + m * M + n];
                if ((k_max - orig) >= thr)
                {
                    // contar caminos k donde min(A[b,m,k], A[b,k,n]) ≈ k_max
                    for (int k = 0; k < M; k++)
                    {
                        float a  = A[b * M * M + m * M + k];
                        float bv = A[b * M * M + k * M + n];
                        float mi = (a < bv) ? a : bv;
                        if (std::abs(mi - k_max) <= MIN_DIFF)
                            count++;
                    }
                }
            }
        }
    }
    return count;
}

// Misma lógica que cpu_maxmin_count pero cuenta solo el argmax por celda,
// igual que assemble_paths (un k por (b,m,n), el primero que alcanza k_max).
int cpu_maxmin_count_argmax(const std::vector<float> &A, int B, int M, float thr)
{
    int count = 0;
    for (int b = 0; b < B; b++)
    {
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < M; n++)
            {
                float k_max  = -1e38f;
                int   best_k = 0;
                for (int k = 0; k < M; k++)
                {
                    float mn = std::min(A[b*M*M + m*M + k], A[b*M*M + k*M + n]);
                    if (mn > k_max) { k_max = mn; best_k = k; }
                }
                (void)best_k;
                float orig = A[b*M*M + m*M + n];
                if ((k_max - orig) >= thr)
                    count++;
            }
        }
    }
    return count;
}

// ─── Test solo GPU (sin referencia CPU) ──────────────────────────────────────
// Para tamaños grandes donde la referencia CPU O(M^3) es inviable en tiempo.
// Solo ejercita el kernel en GPU para fines de profiling con ncu.

void run_gpu_only(int B, int M, float thr)
{
    printf("\n=== [GPU-ONLY] B=%d  M=%d  thr=%.2f ===\n", B, M, thr);

    int blockSize      = 256;
    int total_elements = M * M;
    int gridSize       = (total_elements + blockSize - 1) / blockSize;

    curandState_t *curand_state;
    CHECK_CUDA(cudaMalloc(&curand_state, sizeof(curandState_t) * total_elements));
    randomInit<<<gridSize, blockSize>>>(curand_state, RAND_SEED, total_elements);
    CHECK_CUDA(cudaDeviceSynchronize());

    __half *d_base;
    CHECK_CUDA(cudaMalloc(&d_base, total_elements * sizeof(__half)));
    randomFill<<<gridSize, blockSize>>>(d_base, total_elements, curand_state);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(curand_state));

    std::vector<__half> h_base_half(total_elements);
    CHECK_CUDA(cudaMemcpy(h_base_half.data(), d_base, total_elements * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_base));

    std::vector<__half> h_tensor(B * M * M);
    for (int b = 0; b < B; b++)
        for (int i = 0; i < M * M; i++)
            h_tensor[b * M * M + i] = h_base_half[i];

    TensorResult<__half> t1(MemorySpace::Device, B, M, M);
    TensorResult<__half> t2(MemorySpace::Device, B, M, M);
    CHECK_CUDA(cudaMemcpy(t1.getData(), h_tensor.data(), B * M * M * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(t2.getData(), h_tensor.data(), B * M * M * sizeof(__half), cudaMemcpyHostToDevice));

    __half hthr = __float2half(thr);
    auto result = maxminv4(t1, t2, hthr, 1);
    printf("  effective_order=%d\n", result.effective_order);
}

// ─── Test con validación ─────────────────────────────────────────────────────
// Retorna true si GPU y CPU coinciden en count de paths.

bool run_and_validate(int B, int M, float thr, int n_runs = 3)
{
    printf("\n=== B=%d  M=%d  thr=%.2f ===\n", B, M, thr);

    int blockSize      = 256;
    int total_elements = M * M;
    int gridSize       = (total_elements + blockSize - 1) / blockSize;

    // Inicializar curand con semilla fija para reproducibilidad
    curandState_t *curand_state;
    CHECK_CUDA(cudaMalloc(&curand_state, sizeof(curandState_t) * total_elements));
    randomInit<<<gridSize, blockSize>>>(curand_state, RAND_SEED, total_elements);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Generar datos en device (solo M*M, batch replica el mismo dato)
    __half *d_base;
    CHECK_CUDA(cudaMalloc(&d_base, total_elements * sizeof(__half)));
    randomFill<<<gridSize, blockSize>>>(d_base, total_elements, curand_state);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(curand_state));

    // Copiar datos a host (float) para referencia CPU
    std::vector<__half> h_base_half(total_elements);
    CHECK_CUDA(cudaMemcpy(h_base_half.data(), d_base, total_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    // Construir tensor host [B, M, M] replicando el mismo patrón M×M en cada batch
    std::vector<float> h_A(B * M * M);
    for (int b = 0; b < B; b++)
        for (int i = 0; i < M * M; i++)
            h_A[b * M * M + i] = __half2float(h_base_half[i]);

    CHECK_CUDA(cudaFree(d_base));

    // Referencia CPU (un k por celda, igual que assemble_paths)
    int cpu_count = cpu_maxmin_count_argmax(h_A, B, M, thr);
    printf("  CPU reference (argmax): %d paths\n", cpu_count);

    // Correr GPU n_runs veces y verificar que el resultado sea siempre igual al CPU
    bool all_ok = true;
    int prev_gpu = -1;
    for (int run = 0; run < n_runs; run++)
    {
        // Construir TensorResult device con los datos originales
        TensorResult<__half> t1(MemorySpace::Device, B, M, M);
        TensorResult<__half> t2(MemorySpace::Device, B, M, M);

        // Copiar datos originales (mismos cada run para determinismo de entrada)
        std::vector<__half> h_tensor(B * M * M);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < M * M; i++)
                h_tensor[b * M * M + i] = h_base_half[i];

        CHECK_CUDA(cudaMemcpy(t1.getData(), h_tensor.data(), B * M * M * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(t2.getData(), h_tensor.data(), B * M * M * sizeof(__half), cudaMemcpyHostToDevice));

        __half hthr = __float2half(thr);
        auto result   = maxminv4(t1, t2, hthr, 1);
        // Counter de efectos del orden 1 (misma semántica que cpu_maxmin_count_argmax).
        int gpu_count = result.effects_per_order.empty() ? 0 : result.effects_per_order[0];

        bool match = (gpu_count == cpu_count);
        printf("  Run %d: GPU=%d  %s\n", run, gpu_count, match ? "OK" : "MISMATCH !!!");

        if (!match) all_ok = false;
        if (run > 0 && gpu_count != prev_gpu)
        {
            printf("  *** VARIACION ENTRE RUNS: run%d=%d vs run%d=%d ***\n",
                   run, gpu_count, run - 1, prev_gpu);
            all_ok = false;
        }
        prev_gpu = gpu_count;
    }

    return all_ok;
}

// ─── Test solo GPU (sin referencia CPU) ──────────────────────────────────────
// Una sola corrida del kernel en GPU, pensada para profiling con ncu/nsys:
// un único lanzamiento, sin la referencia CPU O(M^3) que contaminaría el perfil.
void run_gpu_only(int B, int M, float thr)
{
    printf("\n=== [GPU-ONLY] B=%d  M=%d  thr=%.2f ===\n", B, M, thr);

    int blockSize      = 256;
    int total_elements = M * M;
    int gridSize       = (total_elements + blockSize - 1) / blockSize;

    curandState_t *curand_state;
    CHECK_CUDA(cudaMalloc(&curand_state, sizeof(curandState_t) * total_elements));
    randomInit<<<gridSize, blockSize>>>(curand_state, RAND_SEED, total_elements);
    CHECK_CUDA(cudaDeviceSynchronize());

    __half *d_base;
    CHECK_CUDA(cudaMalloc(&d_base, total_elements * sizeof(__half)));
    randomFill<<<gridSize, blockSize>>>(d_base, total_elements, curand_state);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(curand_state));

    std::vector<__half> h_base_half(total_elements);
    CHECK_CUDA(cudaMemcpy(h_base_half.data(), d_base, total_elements * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_base));

    std::vector<__half> h_tensor(B * M * M);
    for (int b = 0; b < B; b++)
        for (int i = 0; i < M * M; i++)
            h_tensor[b * M * M + i] = h_base_half[i];

    TensorResult<__half> t1(MemorySpace::Device, B, M, M);
    TensorResult<__half> t2(MemorySpace::Device, B, M, M);
    CHECK_CUDA(cudaMemcpy(t1.getData(), h_tensor.data(), B * M * M * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(t2.getData(), h_tensor.data(), B * M * M * sizeof(__half),
                          cudaMemcpyHostToDevice));

    __half hthr = __float2half(thr);
    auto result = maxmin(t1, t2, hthr, 1);
    printf("  effective_order=%d\n", result.effective_order);
}

// ─── Main ────────────────────────────────────────────────────────────────────
//
// Uso:
//   fecuda_main                 → suite de validación multi-tamaño (GPU vs CPU).
//   fecuda_main M [B] [thr]     → una sola corrida GPU-only de maxminv2, pensada
//                                 para profiling con ncu/nsys (un kernel, un
//                                 input-size limpio, sin referencia CPU).
int main(int argc, char **argv)
{
    // printf("GPU: sm_75 (GTX 1650)\n");
    printf("Validando MaxMin: comparando GPU vs CPU reference, %d runs por config\n\n", 3);

    // bool all_passed = true;

    // // ── Límites de dimensiones ───────────────────────────────────────────────
    // // M pequeño
    // all_passed &= run_and_validate(1,   4,  0.3f);
    // all_passed &= run_and_validate(1,   8,  0.3f);
    // all_passed &= run_and_validate(1,  16,  0.3f);
    // all_passed &= run_and_validate(1,  32,  0.3f);
    // all_passed &= run_and_validate(1,  64,  0.3f);
    // all_passed &= run_and_validate(1, 128,  0.3f);

    // // ── Variación de threshold ───────────────────────────────────────────────
    // all_passed &= run_and_validate(1, 16, 0.0f);   // thr=0: todos los paths donde k_max > A[m,n]
    // all_passed &= run_and_validate(1, 16, 0.5f);
    // all_passed &= run_and_validate(1, 16, 0.9f);   // thr alto: pocos paths

    // // ── Batch > 1 ────────────────────────────────────────────────────────────
    // all_passed &= run_and_validate(4,  16, 0.3f);
    // all_passed &= run_and_validate(8,  32, 0.3f);
    // all_passed &= run_and_validate(10, 16, 0.4f);  // mismo que test original

    // // ── Batch grande que dispara el camino BATCHED del algoritmo ─────────────
    // // MAX_PATHS_PER_ITER = 100000; B*M*N*K > 100000 → batched path
    // // Con M=64, B=30: 30*64*64*64 = 7,864,320 >> 100000
    // all_passed &= run_and_validate(30, 64, 0.3f);

    // printf("\n==============================\n");
    // printf("Resultado final: %s\n", all_passed ? "TODOS OK" : "HAY FALLOS");
    // printf("==============================\n");

    // return all_passed ? 0 : 1;

    // Rango [100,1000]; mezcla de múltiplos de 32 (256,640,800) y no-múltiplos
    // (100,400,1000) para ejercitar el padding y los tiles de borde.
    bool all_ok = true;
    all_ok &= run_and_validate(1,  90, 0.5f, 1);
    all_ok &= run_and_validate(1,  100, 0.5f, 1);
    all_ok &= run_and_validate(1,  256, 0.5f, 1);
    all_ok &= run_and_validate(1,  400, 0.5f, 1);
    all_ok &= run_and_validate(1,  640, 0.5f, 1);
    all_ok &= run_and_validate(1,  800, 0.5f, 1);
    all_ok &= run_and_validate(1, 1000, 0.5f, 1);

    printf("\n==============================\n");
    printf("Resultado final: %s\n", all_ok ? "TODOS OK" : "HAY FALLOS");
    printf("==============================\n");
    return all_ok ? 0 : 1;
}
