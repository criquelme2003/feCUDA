#include "../../include/core/types.cuh"
#include "../../include/headers.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <vector>

bool g_verbose = true; // definición del global (extern en utils.cuh)

#define MIN_DIFF 0.01f

#define MAX_GRID_SIZE 10000
#define MAX_PATHS_PER_ITER 100000

// Comprueba que una futura allocation no exceda 2GB
// Comprueba que una futura allocation no exceda 2GB y deje 1GB libre en VRAM
static inline bool check_alloc_size_or_fail(size_t bytes, const char *name) {
    const size_t MAX_ALLOC = 22ULL * 1024 * 1024 * 1024; // 22GB
    const size_t RESERVED = 500 * 1024 * 1024ULL;         // 500MB

    if (bytes > MAX_ALLOC) {
        fprintf(stderr, "ERROR: allocation for %s is %zu bytes (>2GB)\n", name, bytes);
        return false;
    }

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

    if (free_bytes < bytes + RESERVED) {
        fprintf(
            stderr,
            "ERROR: not enough free GPU memory for %s: requested=%zubytes free=%zubytes "
            "reserved=%zubytes\n",
            name,
            bytes,
            free_bytes,
            RESERVED
        );
        return false;
    }
    return true;
}

#define CHECK_ALLOC_SIZE_OR_EXIT(bytes, name)                                                      \
    do {                                                                                           \
        if (!check_alloc_size_or_fail((bytes), (name)))                                            \
            exit(EXIT_FAILURE);                                                                    \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Prepend step_order a cada fila (b,m,k,n) y acumula en out_host.
// raw4: buffer host con count*4 ints   layout: [b, m, k, n] por fila
// out_host: vector acumulador con path_width=5
// ─────────────────────────────────────────────────────────────────────────────
// Ensambla o extiende caminos usando el argmax que ya calculó el kernel en GPU.
//
// Primera llamada (prev_paths vacío):
//   Para cada (b,m,n) donde C[m,n] - A[m,n] >= thr, el pivote k viene
//   directamente de argmax[b,m,n]. Devuelve paths [b, m, k, n].
//
// Llamadas siguientes:
//   Para cada path [b,m,...,n], busca n2 donde C[m,n2] - A[m,n2] >= thr
//   y argmax[b,m,n2] == n (el kernel eligió n como pivote óptimo).
//   Devuelve paths extendidos con n2 al final.
//
// A = C_{s-1},  C = C_s,  argmax = resultado del kernel para este step.
using PathsAndValues = std::pair<std::vector<std::vector<int>>, std::vector<float>>;

PathsAndValues assemble_paths(
    std::vector<std::vector<int>> prev_paths,
    __half *d_A,
    __half *d_C,
    int *d_argmax,
    float thr,
    int M,
    int N,
    int B
) {
    std::vector<std::vector<int>> paths;
    std::vector<float>            values;

    int total = M * N * B;
    std::vector<__half> h_A(total);
    std::vector<__half> h_C(total);
    std::vector<int> h_argmax(total);

    cudaMemcpy(h_A.data(),      d_A,      total * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C.data(),      d_C,      total * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_argmax.data(), d_argmax, total * sizeof(int),    cudaMemcpyDeviceToHost);

    auto idx = [&](int b, int m, int n) { return b * M * N + m * N + n; };

    if (!prev_paths.empty()) {
        for (const auto &path : prev_paths) {
            int b = path[0];
            int m = path[1];     // nodo fuente, fijo en todos los steps
            int n = path.back(); // cola actual del camino

            for (int n2 = 0; n2 < N; n2++) {
                int i = idx(b, m, n2);
                float c_val = __half2float(h_C[i]);
                if (c_val - __half2float(h_A[i]) < thr) continue;
                if (h_argmax[i] != n) continue;

                auto new_path = path;
                new_path.push_back(n2);
                paths.push_back(std::move(new_path));
                values.push_back(c_val);
            }
        }
    } else {
        for (int b = 0; b < B; b++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    int i = idx(b, m, n);
                    float c_val = __half2float(h_C[i]);
                    if (c_val - __half2float(h_A[i]) < thr) continue;
                    paths.push_back({b, m, h_argmax[i], n});
                    values.push_back(c_val);
                }
            }
        }
    }
    return {paths, values};
}

MaxminResult maxmin(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order
) {
    if (tensor1.getK() != 1 || tensor2.getK() != 1) {
        printf("Error: maxmin solo acepta tensores 3D (K=1)\n");
        exit(0);
    }

    int B = tensor1.getBatch();
    int M = tensor1.getM();
    int K = tensor1.getN();
    int N = tensor2.getN();

    // Datos en host — B se tipea por columna, A se sube completa una vez
    tensor1.move_to_host();
    tensor2.move_to_host();
    __half *h_A = (__half *)tensor1.getData();
    __half *h_B = (__half *)tensor2.getData();

    LOG(std::cout << "[MAXMIN C++] M=" << M << " B=" << B << " N=" << N << std::endl);
    LOG(std::cout << "[MAXMIN C++] ORDER-" << order << std::endl);

    // Pre-transponer B: [B,K,N] → [N,B,K]
    // La columna n queda contigua en h_B_T + n*B*K  (B*K elementos)
    std::vector<__half> h_B_T((size_t)N * B * K);
    for (int b = 0; b < B; b++)
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                h_B_T[(size_t)n * B * K + b * K + k] = h_B[(size_t)b * K * N + k * N + n];

    // Buffers host
    std::vector<__half> h_C_new((size_t)B * M * N);  // resultado ensamblado [B,M,N]
    std::vector<__half> h_col_tmp((size_t)B * M);     // columna temporal [B,M]

    // Buffers GPU
    __half              *d_C_old, *d_B_col, *d_C_new_col;
    unsigned long long  *d_counter;
    {
        size_t sz_mat = (size_t)B * M * K * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(sz_mat, "d_C_old");
        CHECK_CUDA(cudaMalloc(&d_C_old, sz_mat));
    }
    CHECK_CUDA(cudaMalloc(&d_B_col,     (size_t)B * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C_new_col, (size_t)B * M * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_counter,   sizeof(unsigned long long)));

    // C_old arranca como A
    CHECK_CUDA(cudaMemcpy(d_C_old, h_A, (size_t)B * M * K * sizeof(__half),
                          cudaMemcpyHostToDevice));

    dim3 block(128);
    dim3 grid(M, B);
    size_t shmem = 128 * sizeof(__half);

    int effective_order = 0;
    MaxminResult result;
    result.effects_order1 = 0;

    for (int s = 0; s < order; s++) {
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(unsigned long long)));

        for (int n = 0; n < N; n++) {
            // Subir columna n de B (contigua en h_B_T)
            CHECK_CUDA(cudaMemcpy(d_B_col, &h_B_T[(size_t)n * B * K],
                                  (size_t)B * K * sizeof(__half), cudaMemcpyHostToDevice));

            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                d_C_old, d_B_col, d_C_new_col, d_counter, thr, B, M, K, n);
            CHECK_CUDA(cudaGetLastError());

            // Bajar columna n de C_new (sincroniza el kernel implícitamente)
            CHECK_CUDA(cudaMemcpy(h_col_tmp.data(), d_C_new_col,
                                  (size_t)B * M * sizeof(__half), cudaMemcpyDeviceToHost));

            // Ensamblar en h_C_new [B,M,N]
            for (int b = 0; b < B; b++)
                for (int m = 0; m < M; m++)
                    h_C_new[(size_t)b * M * N + m * N + n] = h_col_tmp[b * M + m];
        }

        unsigned long long h_counter = 0;
        CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        if (h_counter == 0) {
            LOG(std::cout << "[MAXMIN C++] Convergencia en step " << s + 1 << std::endl);
            break;
        }
        LOG(std::cout << "[MAXMIN C++] Efectos encontrados en orden " << s + 1
                      << " : " << h_counter << std::endl);

        effective_order = s + 1;
        if (s == 0) result.effects_order1 = h_counter;

        // C_new → C_old para el siguiente step
        CHECK_CUDA(cudaMemcpy(d_C_old, h_C_new.data(),
                              (size_t)B * M * N * sizeof(__half), cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaFree(d_C_old));
    CHECK_CUDA(cudaFree(d_B_col));
    CHECK_CUDA(cudaFree(d_C_new_col));
    CHECK_CUDA(cudaFree(d_counter));

    result.effective_order = effective_order;
    return result;
}
