#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>


// Macro para verificar errores de CUDA con exceptions
#define CHECK_CUDA(call)                                                                           \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::string error_msg = std::string("CUDA error at ") + __FILE__ + ":" +               \
                                    std::to_string(__LINE__) + ": " + cudaGetErrorString(err);     \
            std::cerr << error_msg << std::endl;                                                   \
            cudaDeviceReset();                                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

#define CHECK_KERNEL()                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__ << " : "    \
                      << cudaGetErrorString(err) << std::endl;                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

#define CHECK_CURAND(x)                                                                            \
    do {                                                                                           \
        if ((x) != CURAND_STATUS_SUCCESS) {                                                        \
            printf("Error at %s:%d\n", __FILE__, __LINE__);                                        \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (0)

// Comprueba que una futura allocation no deje la VRAM ocupada por encima de
// max_vram_fraction (0..1) del total detectado vía cudaMemGetInfo.
static inline bool
check_alloc_size_or_fail(size_t bytes, const char *name, float max_vram_fraction = 0.7f) {
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "WARN: cudaMemGetInfo failed (%s). Allowing allocation for %s of %zu bytes\n",
            cudaGetErrorString(err),
            name,
            bytes
        );
        return true; // no info available → be permissive (preserves previous behaviour)
    }

    size_t max_occupied = (size_t)(max_vram_fraction * (double)total_bytes);
    size_t occupied_after = (total_bytes - free_bytes) + bytes;

    if (occupied_after > max_occupied) {
        fprintf(
            stderr,
            "ERROR: allocation for %s would occupy %zu bytes (> %.1f%% of %zu bytes total "
            "VRAM = %zu bytes budget; currently used=%zu free=%zu)\n",
            name,
            occupied_after,
            max_vram_fraction * 100.0f,
            total_bytes,
            max_occupied,
            total_bytes - free_bytes,
            free_bytes
        );
        return false;
    }
    return true;
}

#define CHECK_ALLOC_SIZE_OR_EXIT(bytes, name)                                                     \
    do {                                                                                           \
        if (!check_alloc_size_or_fail((bytes), (name)))                                           \
            exit(EXIT_FAILURE);                                                                   \
    } while (0)

#define CHECK_ALLOC_SIZE_OR_EXIT_PCT(bytes, name, max_vram_fraction)                              \
    do {                                                                                           \
        if (!check_alloc_size_or_fail((bytes), (name), (max_vram_fraction)))                      \
            exit(EXIT_FAILURE);                                                                   \
    } while (0)


// Estima la VRAM (en bytes) necesaria para ejecutar cualquiera de los maxmin
static inline size_t
estimate_maxmin_vram_bytes(int B, int M, int K, int N, int padding = 64) {
    if (padding < 1) padding = 1;

    auto round_up_to = [](long long v, long long multiple) -> long long {
        return ((v + multiple - 1) / multiple) * multiple;
    };

    long long Mpad = round_up_to(M, padding);
    long long Kpad = round_up_to(K, padding);
    long long Npad = round_up_to(N, padding);

    long long a_elems = (long long)B * Mpad * Kpad;
    long long b_elems = (long long)B * Kpad * Npad;
    long long c_elems = (long long)B * Mpad * Npad; // C_dev_before / C_dev_after / argmax

    size_t bytes = 0;
    bytes += (size_t)a_elems * sizeof(__half);       // d_A
    bytes += (size_t)b_elems * sizeof(__half);       // d_B
    bytes += (size_t)c_elems * sizeof(__half) * 2;   // C_dev_before + C_dev_after
    bytes += (size_t)c_elems * sizeof(int);          // argmax
    bytes += sizeof(int);                            // d_counter

    return bytes;
}

// ── Global verbose flag ──────────────────────────────────────────────────────
extern bool g_verbose;
#define LOG(x)                                                                                     \
    do {                                                                                           \
        if (g_verbose) {                                                                           \
            x;                                                                                     \
        }                                                                                          \
    } while (0)

#endif
