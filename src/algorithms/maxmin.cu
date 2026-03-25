#include "../../include/core/types.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <vector>

bool g_verbose = true;  // definición del global (extern en utils.cuh)

#define MAX_GRID_SIZE    10000
#define MAX_PATHS_PER_ITER 100000

// Comprueba que una futura allocation no exceda 2GB
// Comprueba que una futura allocation no exceda 2GB y deje 1GB libre en VRAM
static inline bool check_alloc_size_or_fail(size_t bytes, const char *name) {
    const size_t MAX_ALLOC = 3.5 * 1024 * 1024 * 1024ULL; // 3.5GB
    const size_t RESERVED = 500 * 1024 * 1024ULL ; // 500MB
    
    if (bytes > MAX_ALLOC) {
        fprintf(stderr, "ERROR: allocation for %s is %zu bytes (>2GB)\n", name, bytes);
        return false;
    }

    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "WARN: cudaMemGetInfo failed (%s). Allowing allocation for %s of %zu bytes\n",
                cudaGetErrorString(err), name, bytes);
        return true; // no info available → be permissive (preserves previous behaviour)
    }

    if (free_bytes < bytes + RESERVED ) {
        fprintf(stderr,
                "ERROR: not enough free GPU memory for %s: requested=%zubytes free=%zubytes reserved=%zubytes\n",
                name, bytes, free_bytes, RESERVED);
        return false;
    }
    return true;
}

#define CHECK_ALLOC_SIZE_OR_EXIT(bytes, name) \
    do { if (!check_alloc_size_or_fail((bytes),(name))) exit(EXIT_FAILURE); } while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Prepend step_order a cada fila (b,m,k,n) y acumula en out_host.
// raw4: buffer host con count*4 ints   layout: [b, m, k, n] por fila
// out_host: vector acumulador con path_width=5
// ─────────────────────────────────────────────────────────────────────────────
static void prepend_order(const int* raw4, int count, int step_order,
                          std::vector<int>& out_host)
{
    for (int i = 0; i < count; i++) {
        out_host.push_back(step_order);
        out_host.push_back(raw4[i * 4 + 0]); // b
        out_host.push_back(raw4[i * 4 + 1]); // m
        out_host.push_back(raw4[i * 4 + 2]); // k
        out_host.push_back(raw4[i * 4 + 3]); // n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::tuple<int*, __half*, int, int, int>>
maxmin(TensorResult<__half> &tensor1, TensorResult<__half> &tensor2,
       __half thr, int order, bool return_paths)
{
    std::vector<std::tuple<int*, __half*, int, int, int>> ret;

    if (tensor1.getK() != 1 || tensor2.getK() != 1) {
        printf("Error: maxmin solo acepta tensores 3D (K=1)\n");
        exit(0);
    }

    // GET NECESARY DIMENSIONS
    int B = tensor1.getBatch();
    int M = tensor1.getM();
    int K = tensor1.getN(); // N del tensor1 actúa como K en el kernel
    int N = tensor2.getN();
    int total_elems = B * M * N;

    tensor1.move_to_device();
    tensor2.move_to_device();
    __half* d_A = (__half*)tensor1.getData();
    __half* d_B = (__half*)tensor2.getData();


    // Configuración de lanzamiento (fija para todos los paths)
    int blockDim = 128;
    dim3 block(blockDim); // define septs for thread loop
    size_t shmem = blockDim * (sizeof(__half) + sizeof(int)); // shared memory save k_values and k_index for block reduction

    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN 1: un solo paso A⊗B
    // ─────────────────────────────────────────────────────────────────────────
    if (order == 1)
    {

      // alloc fot c mtx
        __half* C_out;
        {
            size_t __alloc_bytes = (size_t)total_elems * sizeof(__half);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_out");
            CHECK_CUDA(cudaMalloc(&C_out, __alloc_bytes));
        }

        int*    d_raw_paths;
        __half* d_raw_values;
        int     h_count = 0;
        int*    d_counter;

      // alloc ant init  path counter
        {
            size_t __alloc_bytes = (size_t)sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_counter");
            CHECK_CUDA(cudaMalloc(&d_counter, __alloc_bytes));
        }
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

      // alloc  paths array
        {
            size_t __alloc_bytes = (size_t)MAX_PATHS_PER_ITER * 4 * sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_raw_paths");
            CHECK_CUDA(cudaMalloc(&d_raw_paths, __alloc_bytes));
        }
      // alloc values array
        {
            size_t __alloc_bytes = (size_t)MAX_PATHS_PER_ITER * sizeof(__half);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_raw_values");
            CHECK_CUDA(cudaMalloc(&d_raw_values, __alloc_bytes));
        }


        if ((long long)B * M * N * K < MAX_PATHS_PER_ITER)
        {
            LOG(std::cout << "[MAXMIN C++] ORDER-1 COMPLETE" << std::endl);
            dim3 grid(N, M, B);
            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                d_A, d_B, C_out,
                d_raw_paths, d_raw_values, d_counter,
                /*argmax=*/nullptr,
                thr, B, M, N, K, -1, -1);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
        }
        else
        {
            LOG(std::cout << "[MAXMIN C++] ORDER-1 BATCHED" << std::endl);
            int sizeA = M * K, sizeB = K * N;
            dim3 grid(N, M, 1);
            std::vector<int*>    d_paths_acc;
            std::vector<__half*> d_values_acc;
            std::vector<int>     h_counts_acc;

            int* d_ctr_b;
            {
                size_t __alloc_bytes = (size_t)sizeof(int);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_ctr_b");
                CHECK_CUDA(cudaMalloc(&d_ctr_b, __alloc_bytes));
            }

            for (int b_ = 0; b_ < B; b_++)
            {
                int*    dp;
                __half* dv;
                int     mppi = M * K * N;
                {
                    size_t __alloc_bytes = (size_t)mppi * 4 * sizeof(int);
                    CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "dp");
                    CHECK_CUDA(cudaMalloc(&dp, __alloc_bytes));
                }
                {
                    size_t __alloc_bytes = (size_t)mppi * sizeof(__half);
                    CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "dv");
                    CHECK_CUDA(cudaMalloc(&dv, __alloc_bytes));
                }
                CHECK_CUDA(cudaMemset(d_ctr_b, 0, sizeof(int)));

                maxmin_threshold_kernel<<<grid, block, shmem>>>(
                    d_A + b_ * sizeA,
                    d_B + b_ * sizeB,
                    C_out + b_ * M * N,
                    dp, dv, d_ctr_b,
                    /*argmax=*/nullptr,
                    thr, 1, M, N, K, 0, -1);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());

                int cnt = 0;
                CHECK_CUDA(cudaMemcpy(&cnt, d_ctr_b, sizeof(int), cudaMemcpyDeviceToHost));
                d_paths_acc.push_back(dp);
                d_values_acc.push_back(dv);
                h_counts_acc.push_back(cnt);
                h_count += cnt;
            }
                CHECK_CUDA(cudaFree(d_ctr_b));

            // Reusar d_raw_paths / d_raw_values para el merge
            CHECK_CUDA(cudaFree(d_raw_paths));
            CHECK_CUDA(cudaFree(d_raw_values));
            {
                size_t __alloc_bytes = (size_t)h_count * 4 * sizeof(int);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_raw_paths");
                CHECK_CUDA(cudaMalloc(&d_raw_paths, __alloc_bytes));
            }
            {
                size_t __alloc_bytes = (size_t)h_count * sizeof(__half);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_raw_values");
                CHECK_CUDA(cudaMalloc(&d_raw_values, __alloc_bytes));
            }

            int offset = 0;
            for (int c = 0; c < B; c++) {
                int cnt = h_counts_acc[c];
                if (cnt > 0) {
                    CHECK_CUDA(cudaMemcpy(d_raw_paths  + offset * 4,
                        d_paths_acc[c],  (size_t)cnt * 4 * sizeof(int),
                        cudaMemcpyDeviceToDevice));
                    CHECK_CUDA(cudaMemcpy(d_raw_values + offset,
                        d_values_acc[c], (size_t)cnt * sizeof(__half),
                        cudaMemcpyDeviceToDevice));
                    offset += cnt;
                }
                CHECK_CUDA(cudaFree(d_paths_acc[c]));
                CHECK_CUDA(cudaFree(d_values_acc[c]));
            }
        }
        CHECK_CUDA(cudaFree(C_out));
        CHECK_CUDA(cudaFree(d_counter));
        LOG(std::cout << "[MAXMIN C++] ORDER-1 paths found: " << h_count << std::endl);

        // Prepend order=1 en CPU → (order, b, m, k, n)
        int path_width = 5;
        int*   d_final_paths  = nullptr;
        __half* d_final_values = nullptr;

        if (h_count > 0) {
            std::vector<int>    h_raw4((size_t)h_count * 4);
            std::vector<__half> h_vals((size_t)h_count);
            CHECK_CUDA(cudaMemcpy(h_raw4.data(), d_raw_paths,
                (size_t)h_count * 4 * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_vals.data(), d_raw_values,
                (size_t)h_count * sizeof(__half), cudaMemcpyDeviceToHost));

            std::vector<int> h_out;
            h_out.reserve((size_t)h_count * 5);
            prepend_order(h_raw4.data(), h_count, 1, h_out);

            {
                size_t __alloc_bytes = (size_t)h_count * 5 * sizeof(int);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_final_paths");
                CHECK_CUDA(cudaMalloc(&d_final_paths, __alloc_bytes));
            }
            {
                size_t __alloc_bytes = (size_t)h_count * sizeof(__half);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_final_values");
                CHECK_CUDA(cudaMalloc(&d_final_values, __alloc_bytes));
            }
            CHECK_CUDA(cudaMemcpy(d_final_paths, h_out.data(),
                (size_t)h_count * 5 * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_final_values, h_vals.data(),
                (size_t)h_count * sizeof(__half), cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaFree(d_raw_paths));
        CHECK_CUDA(cudaFree(d_raw_values));

        ret.push_back(std::make_tuple(d_final_paths, d_final_values,
                                      h_count, path_width, 1));
        return ret;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN > 1: iterativo con el mismo kernel
    // ─────────────────────────────────────────────────────────────────────────
    LOG(std::cout << "[MAXMIN C++] ORDER-" << order << " return_paths="
              << (return_paths ? "true" : "false") << std::endl);

    // Buffers iterativos: C_dev_before (= C_prev) y C_dev_after (= C_next)
    __half *C_dev_before, *C_dev_after;
    {
        size_t __alloc_bytes = (size_t)total_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_before");
        CHECK_CUDA(cudaMalloc(&C_dev_before, __alloc_bytes));
    }
    {
        size_t __alloc_bytes = (size_t)total_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_after");
        CHECK_CUDA(cudaMalloc(&C_dev_after, __alloc_bytes));
    }
    // C_dev_before arranca como copia de d_A
    CHECK_CUDA(cudaMemcpy(C_dev_before, d_A,
        total_elems * sizeof(__half), cudaMemcpyDeviceToDevice));

    dim3 grid(N, M, B);
    float thr_f = __half2float(thr);
    int effective_order = 0;

    // ─────────────────────────────────────────────────────────────────────────
    // return_paths = false: emitir aristas crudas por step
    // ─────────────────────────────────────────────────────────────────────────
    if (!return_paths)
    {
        int* d_counter;
        {
            size_t __alloc_bytes = (size_t)sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_counter");
            CHECK_CUDA(cudaMalloc(&d_counter, __alloc_bytes));
        }

        std::vector<int>    h_all_paths;
        std::vector<__half> h_all_values;

        for (int s = 0; s < order; s++)
        {
            // ── Pasada 1: solo contar (paths=nullptr) ────────────────────────
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                C_dev_before, d_B, C_dev_after,
                /*paths=*/nullptr, /*values=*/nullptr, d_counter,
                /*argmax=*/nullptr,
                thr, B, M, N, K, -1, -1);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            int step_count = 0;
            CHECK_CUDA(cudaMemcpy(&step_count, d_counter, sizeof(int),
                                  cudaMemcpyDeviceToHost));

            if (step_count == 0) {
                LOG(std::cout << "[MAXMIN C++] Convergencia en step " << s + 1
                          << " (sin nuevas aristas)" << std::endl);
                break;
            }

            // ── Pasada 2: alocar exacto y emitir ────────────────────────────
            int*    d_paths_step;
            __half* d_values_step;
            {
                size_t __alloc_bytes = (size_t)step_count * 4 * sizeof(int);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_paths_step");
                CHECK_CUDA(cudaMalloc(&d_paths_step, __alloc_bytes));
            }
            {
                size_t __alloc_bytes = (size_t)step_count * sizeof(__half);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_values_step");
                CHECK_CUDA(cudaMalloc(&d_values_step, __alloc_bytes));
            }
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                C_dev_before, d_B, C_dev_after,
                d_paths_step, d_values_step, d_counter,
                /*argmax=*/nullptr,
                thr, B, M, N, K, -1, -1);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            std::vector<int>    h_raw4((size_t)step_count * 4);
            std::vector<__half> h_vals((size_t)step_count);
            CHECK_CUDA(cudaMemcpy(h_raw4.data(), d_paths_step,
                (size_t)step_count * 4 * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_vals.data(), d_values_step,
                (size_t)step_count * sizeof(__half), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaFree(d_paths_step));
            CHECK_CUDA(cudaFree(d_values_step));

            prepend_order(h_raw4.data(), step_count, s + 1, h_all_paths);
            h_all_values.insert(h_all_values.end(), h_vals.begin(), h_vals.end());

            effective_order = s + 1;
            std::swap(C_dev_before, C_dev_after);
        }

        CHECK_CUDA(cudaFree(C_dev_before));
        CHECK_CUDA(cudaFree(C_dev_after));
        CHECK_CUDA(cudaFree(d_counter));

        int total_count = (int)h_all_values.size();
        LOG(std::cout << "[MAXMIN C++] Total aristas (return_paths=false): "
                  << total_count << "  effective_order=" << effective_order << std::endl);

        int*   d_out_paths  = nullptr;
        __half* d_out_values = nullptr;
        if (total_count > 0) {
            {
                size_t __alloc_bytes = (size_t)total_count * 5 * sizeof(int);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_out_paths");
                CHECK_CUDA(cudaMalloc(&d_out_paths, __alloc_bytes));
            }
            {
                size_t __alloc_bytes = (size_t)total_count * sizeof(__half);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_out_values");
                CHECK_CUDA(cudaMalloc(&d_out_values, __alloc_bytes));
            }
            CHECK_CUDA(cudaMemcpy(d_out_paths, h_all_paths.data(),
                (size_t)total_count * 5 * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_out_values, h_all_values.data(),
                (size_t)total_count * sizeof(__half), cudaMemcpyHostToDevice));
        }
        ret.push_back(std::make_tuple(d_out_paths, d_out_values,
                                      total_count, 5, effective_order));
        return ret;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // return_paths = true: reconstrucción de caminos completos
    // ─────────────────────────────────────────────────────────────────────────
    // Advertencia si argmax excedería 2 GB de VRAM
    {
        size_t argmax_total = (size_t)order * total_elems * sizeof(int);
        if (argmax_total > 2ULL * 1024 * 1024 * 1024) {
            printf("[MAXMIN WARNING] return_paths=true con N=%d order=%d requiere "
                   "~%zu MB para argmax. Considera return_paths=false.\n",
                   M, order, argmax_total / (1024 * 1024));
        }
    }

    // Convergencia en GPU: solo 4 bytes D2H por step (antes: total_elems*2 bytes)
    int* d_new_count;
    {
        size_t __alloc_bytes = (size_t)sizeof(int);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_new_count");
        CHECK_CUDA(cudaMalloc(&d_new_count, __alloc_bytes));
    }

    // Para reconstrucción: guardar C antes y después del último step efectivo
    std::vector<__half> h_C_final    (total_elems);
    std::vector<__half> h_C_pre_final(total_elems);

    // Inicializar h_C_pre_final con A_orig (step 0 = identity)
    CHECK_CUDA(cudaMemcpy(h_C_pre_final.data(), d_A,
        total_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    std::vector<int*> d_argmax;
    d_argmax.reserve(order);

    dim3 grid_r(std::min((total_elems + 255) / 256, 1024));
    dim3 block_r(256);
    size_t shmem_r = 256 * sizeof(int);

    for (int s = 0; s < order; s++)
    {
        int* d_am;
        {
            size_t __alloc_bytes = (size_t)total_elems * sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_am");
            CHECK_CUDA(cudaMalloc(&d_am, __alloc_bytes));
        }
        d_argmax.push_back(d_am);

        maxmin_threshold_kernel<<<grid, block, shmem>>>(
            C_dev_before, d_B, C_dev_after,
            /*paths=*/nullptr, /*values=*/nullptr, /*counter=*/nullptr,
            d_am,
            thr, B, M, N, K, -1, -1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Convergencia: contar en GPU (4 bytes D2H)
        CHECK_CUDA(cudaMemset(d_new_count, 0, sizeof(int)));
        count_new_kernel<<<grid_r, block_r, shmem_r>>>(
            C_dev_before, C_dev_after, d_new_count, thr_f, total_elems);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        int new_count = 0;
        CHECK_CUDA(cudaMemcpy(&new_count, d_new_count, sizeof(int),
                              cudaMemcpyDeviceToHost));

        if (new_count > 0) {
            effective_order = s + 1;
            // Guardar C_before (= C_{s}) y C_after (= C_{s+1}) para reconstrucción
            CHECK_CUDA(cudaMemcpy(h_C_pre_final.data(), C_dev_before,
                total_elems * sizeof(__half), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_C_final.data(), C_dev_after,
                total_elems * sizeof(__half), cudaMemcpyDeviceToHost));
            std::swap(C_dev_before, C_dev_after);
        } else {
            LOG(std::cout << "[MAXMIN C++] Convergencia en step " << s + 1
                      << " (sin nuevos caminos)" << std::endl);
            CHECK_CUDA(cudaFree(d_argmax.back()));
            d_argmax.pop_back();
            break;
        }
    }

    CHECK_CUDA(cudaFree(d_new_count));
    CHECK_CUDA(cudaFree(C_dev_before));
    CHECK_CUDA(cudaFree(C_dev_after));

    int n_steps = (int)d_argmax.size();  // = effective_order

    // Traer argmax a CPU
    std::vector<std::vector<int>> h_argmax(n_steps, std::vector<int>(total_elems));
    for (int s = 0; s < n_steps; s++) {
        CHECK_CUDA(cudaMemcpy(h_argmax[s].data(), d_argmax[s],
            total_elems * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_argmax[s]));
    }

    // ── Reconstrucción de caminos ────────────────────────────────────────────
    // path: [effective_order, b, m, k0, k1, ..., k_{n_steps-1}, n]
    // width = n_steps + 4
    int path_width = n_steps + 4;
    std::vector<int>    h_paths;
    std::vector<__half> h_values;

    for (int b_ = 0; b_ < B; b_++)
    {
        for (int m_ = 0; m_ < M; m_++)
        {
            for (int n_ = 0; n_ < N; n_++)
            {
                int idx = b_ * M * N + m_ * N + n_;
                // Par nuevo en el último step efectivo
                float val_after  = __half2float(h_C_final[idx]);
                float val_before = __half2float(h_C_pre_final[idx]);
                if ((val_after - val_before) < thr_f) continue;

                // Reconstrucción hacia atrás
                std::vector<int> ks(n_steps);
                int cur = n_;
                for (int s = n_steps - 1; s >= 0; s--) {
                    ks[s] = h_argmax[s][b_ * M * N + m_ * N + cur];
                    cur   = ks[s];
                }

                h_paths.push_back(effective_order);
                h_paths.push_back(b_);
                h_paths.push_back(m_);
                for (int k : ks) h_paths.push_back(k);
                h_paths.push_back(n_);
                h_values.push_back(h_C_final[idx]);
            }
        }
    }

    int total_count = (int)h_values.size();
    LOG(std::cout << "[MAXMIN C++] Caminos reconstruidos (effective_order="
              << effective_order << "): " << total_count << std::endl);

    int*   d_out_paths  = nullptr;
    __half* d_out_values = nullptr;
    if (total_count > 0) {
        {
            size_t __alloc_bytes = (size_t)total_count * path_width * sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_out_paths");
            CHECK_CUDA(cudaMalloc(&d_out_paths, __alloc_bytes));
        }
        {
            size_t __alloc_bytes = (size_t)total_count * sizeof(__half);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_out_values");
            CHECK_CUDA(cudaMalloc(&d_out_values, __alloc_bytes));
        }
        CHECK_CUDA(cudaMemcpy(d_out_paths, h_paths.data(),
            (size_t)total_count * path_width * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_out_values, h_values.data(),
            (size_t)total_count * sizeof(__half), cudaMemcpyHostToDevice));
    }

    ret.push_back(std::make_tuple(d_out_paths, d_out_values,
                                  total_count, path_width, effective_order));
    return ret;
}

// ─────────────────────────────────────────────────────────────────────────────
// maxmin_reduced — idéntica a maxmin pero para grafos esparsos:
//   return_paths=false: single-pass con buffer de (avg_n + 1) * N aristas por step
//                       (en lugar de two-pass). Si se excede el buffer se trunca.
//   return_paths=true:  idéntico a maxmin.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::tuple<int*, __half*, int, int, int>>
maxmin_reduced(TensorResult<__half> &tensor1, TensorResult<__half> &tensor2,
               __half thr, int order, bool return_paths, float avg_n)
{
  printf("Executing maxmin reduced\n");
  std::vector<std::tuple<int*, __half*, int, int, int>> ret;

    if (tensor1.getK() != 1 || tensor2.getK() != 1) {
        printf("Error: maxmin_reduced solo acepta tensores 3D (K=1)\n");
        exit(0);
    }
// Get dims
    int B = tensor1.getBatch();
    int M = tensor1.getM();
    int K = tensor1.getN();
    int N = tensor2.getN();
    int total_elems = B * M * N;

    tensor1.move_to_device();
    tensor2.move_to_device();
    __half* d_A = (__half*)tensor1.getData();
    __half* d_B = (__half*)tensor2.getData();

    dim3 block(128);
    size_t shmem = 128 * (sizeof(__half) + sizeof(int));

    LOG(std::cout << "[MAXMIN_REDUCED] avg_n=" << avg_n
              << " order=" << order << " N=" << N << std::endl);

// aloc for c mtxs
    __half *C_dev_before, *C_dev_after;
    {
        size_t __alloc_bytes = (size_t)total_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_before_reduced");
        CHECK_CUDA(cudaMalloc(&C_dev_before, __alloc_bytes));
    }
    {
        size_t __alloc_bytes = (size_t)total_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_after_reduced");
        CHECK_CUDA(cudaMalloc(&C_dev_after, __alloc_bytes));
    }
    // initial copy, for first iteration
    CHECK_CUDA(cudaMemcpy(C_dev_before, d_A,
        total_elems * sizeof(__half), cudaMemcpyDeviceToDevice));


    // grid dims. with N = M = n = 10000, B = 1,  10⁸ blocks
    dim3 grid(N, M, B);
    float thr_f       = __half2float(thr);
    int   pre_alloc   = (int)((avg_n + 1.0f) * (float)N); // prealloc for paths
    int   effective_order = 0;

    // ─────────────────────────────────────────────────────────────────────────
    // return_paths = false: single-pass con buffer fijo de pre_alloc aristas
    // ─────────────────────────────────────────────────────────────────────────
    if (!return_paths)
    {
        int*    d_counter;
        int*    d_paths_step;
        __half* d_vals_step;
        {
            size_t __alloc_bytes = (size_t)sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_counter_reduced");
            CHECK_CUDA(cudaMalloc(&d_counter, __alloc_bytes));
        }
        {
            size_t __alloc_bytes = (size_t)pre_alloc * 4 * sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_paths_step_reduced");
            CHECK_CUDA(cudaMalloc(&d_paths_step, __alloc_bytes));
        }
        {
            size_t __alloc_bytes = (size_t)pre_alloc * sizeof(__half);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_vals_step_reduced");
            CHECK_CUDA(cudaMalloc(&d_vals_step, __alloc_bytes));
        }

        std::vector<int>    h_all_paths;
        std::vector<__half> h_all_values;

        for (int s = 0; s < order; s++)
        {
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                C_dev_before, d_B, C_dev_after,
                d_paths_step, d_vals_step, d_counter,
                /*argmax=*/nullptr,
                thr, B, M, N, K, -1, pre_alloc);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            int step_count = 0;
            CHECK_CUDA(cudaMemcpy(&step_count, d_counter, sizeof(int),
                                  cudaMemcpyDeviceToHost));

            if (step_count == 0) {
                LOG(std::cout << "[MAXMIN_REDUCED] Convergencia en step " << s + 1
                          << " (sin nuevas aristas)" << std::endl);
                break;
            }

            if (step_count > pre_alloc) {
                LOG(printf("[MAXMIN_REDUCED] step %d: buffer (%d) excedido, "
                           "real=%d. Aumentar avg_n.\n", s + 1, pre_alloc, step_count));
                step_count = pre_alloc; // truncar: solo emitimos las primeras pre_alloc
            }

            std::vector<int>    h_raw4((size_t)step_count * 4);
            std::vector<__half> h_vals((size_t)step_count);
            CHECK_CUDA(cudaMemcpy(h_raw4.data(), d_paths_step,
                (size_t)step_count * 4 * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_vals.data(), d_vals_step,
                (size_t)step_count * sizeof(__half), cudaMemcpyDeviceToHost));

            prepend_order(h_raw4.data(), step_count, s + 1, h_all_paths);
            h_all_values.insert(h_all_values.end(), h_vals.begin(), h_vals.end());

            effective_order = s + 1;
            std::swap(C_dev_before, C_dev_after);
        }

        CHECK_CUDA(cudaFree(d_paths_step));
        CHECK_CUDA(cudaFree(d_vals_step));
        CHECK_CUDA(cudaFree(d_counter));
        CHECK_CUDA(cudaFree(C_dev_before));
        CHECK_CUDA(cudaFree(C_dev_after));

        int total_count = (int)h_all_values.size();
        LOG(std::cout << "[MAXMIN_REDUCED] Total aristas: "
                  << total_count << "  effective_order=" << effective_order << std::endl);

        int*    d_out_paths  = nullptr;
        __half* d_out_values = nullptr;
        if (total_count > 0) {
            {
                size_t __alloc_bytes = (size_t)total_count * 5 * sizeof(int);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_out_paths_reduced");
                CHECK_CUDA(cudaMalloc(&d_out_paths, __alloc_bytes));
            }
            {
                size_t __alloc_bytes = (size_t)total_count * sizeof(__half);
                CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_out_values_reduced");
                CHECK_CUDA(cudaMalloc(&d_out_values, __alloc_bytes));
            }
            CHECK_CUDA(cudaMemcpy(d_out_paths, h_all_paths.data(),
                (size_t)total_count * 5 * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_out_values, h_all_values.data(),
                (size_t)total_count * sizeof(__half), cudaMemcpyHostToDevice));
        }
        ret.push_back(std::make_tuple(d_out_paths, d_out_values,
                                      total_count, 5, effective_order));
        return ret;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // return_paths = true: idéntico a maxmin (reconstrucción completa)
    // ─────────────────────────────────────────────────────────────────────────
    {
        size_t argmax_total = (size_t)order * total_elems * sizeof(int);
        if (argmax_total > 2ULL * 1024 * 1024 * 1024) {
            printf("[MAXMIN_REDUCED WARNING] return_paths=true con N=%d order=%d requiere "
                   "~%zu MB para argmax. Considera return_paths=false.\n",
                   M, order, argmax_total / (1024 * 1024));
        }
    }

    int* d_new_count;
    {
        size_t __alloc_bytes = (size_t)sizeof(int);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_new_count_reduced");
        CHECK_CUDA(cudaMalloc(&d_new_count, __alloc_bytes));
    }

    std::vector<__half> h_C_final    (total_elems);
    std::vector<__half> h_C_pre_final(total_elems);
    CHECK_CUDA(cudaMemcpy(h_C_pre_final.data(), d_A,
        total_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    std::vector<int*> d_argmax;
    d_argmax.reserve(order);

    dim3 grid_r(std::min((total_elems + 255) / 256, 1024));
    dim3 block_r(256);
    size_t shmem_r = 256 * sizeof(int);

    for (int s = 0; s < order; s++)
    {
        int* d_am;
        {
            size_t __alloc_bytes = (size_t)total_elems * sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_am_reduced");
            CHECK_CUDA(cudaMalloc(&d_am, __alloc_bytes));
        }
        d_argmax.push_back(d_am);

        maxmin_threshold_kernel<<<grid, block, shmem>>>(
            C_dev_before, d_B, C_dev_after,
            /*paths=*/nullptr, /*values=*/nullptr, /*counter=*/nullptr,
            d_am,
            thr, B, M, N, K, -1, -1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_new_count, 0, sizeof(int)));
        count_new_kernel<<<grid_r, block_r, shmem_r>>>(
            C_dev_before, C_dev_after, d_new_count, thr_f, total_elems);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        int new_count = 0;
        CHECK_CUDA(cudaMemcpy(&new_count, d_new_count, sizeof(int),
                              cudaMemcpyDeviceToHost));

        if (new_count > 0) {
            effective_order = s + 1;
            CHECK_CUDA(cudaMemcpy(h_C_pre_final.data(), C_dev_before,
                total_elems * sizeof(__half), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_C_final.data(), C_dev_after,
                total_elems * sizeof(__half), cudaMemcpyDeviceToHost));
            std::swap(C_dev_before, C_dev_after);
        } else {
            LOG(std::cout << "[MAXMIN_REDUCED] Convergencia en step " << s + 1
                      << " (sin nuevos caminos)" << std::endl);
            CHECK_CUDA(cudaFree(d_argmax.back()));
            d_argmax.pop_back();
            break;
        }
    }

    CHECK_CUDA(cudaFree(d_new_count));
    CHECK_CUDA(cudaFree(C_dev_before));
    CHECK_CUDA(cudaFree(C_dev_after));

    int n_steps    = (int)d_argmax.size();
    int path_width = n_steps + 4;
    std::vector<int>    h_paths;
    std::vector<__half> h_values;

    std::vector<std::vector<int>> h_argmax(n_steps, std::vector<int>(total_elems));
    for (int s = 0; s < n_steps; s++) {
        CHECK_CUDA(cudaMemcpy(h_argmax[s].data(), d_argmax[s],
            total_elems * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_argmax[s]));
    }

    for (int b_ = 0; b_ < B; b_++)
        for (int m_ = 0; m_ < M; m_++)
            for (int n_ = 0; n_ < N; n_++) {
                int idx = b_ * M * N + m_ * N + n_;
                float val_after  = __half2float(h_C_final[idx]);
                float val_before = __half2float(h_C_pre_final[idx]);
                if ((val_after - val_before) < thr_f) continue;

                std::vector<int> ks(n_steps);
                int cur = n_;
                for (int s = n_steps - 1; s >= 0; s--) {
                    ks[s] = h_argmax[s][b_ * M * N + m_ * N + cur];
                    cur   = ks[s];
                }
                h_paths.push_back(effective_order);
                h_paths.push_back(b_);
                h_paths.push_back(m_);
                for (int k : ks) h_paths.push_back(k);
                h_paths.push_back(n_);
                h_values.push_back(h_C_final[idx]);
            }

    int total_count = (int)h_values.size();
    LOG(std::cout << "[MAXMIN_REDUCED] Caminos reconstruidos (effective_order="
              << effective_order << "): " << total_count << std::endl);

    int*    d_out_paths  = nullptr;
    __half* d_out_values = nullptr;
    if (total_count > 0) {
        {
            size_t __alloc_bytes = (size_t)total_count * path_width * sizeof(int);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_out_paths_reduced");
            CHECK_CUDA(cudaMalloc(&d_out_paths, __alloc_bytes));
        }
        {
            size_t __alloc_bytes = (size_t)total_count * sizeof(__half);
            CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "d_out_values_reduced");
            CHECK_CUDA(cudaMalloc(&d_out_values, __alloc_bytes));
        }
        CHECK_CUDA(cudaMemcpy(d_out_paths, h_paths.data(),
            (size_t)total_count * path_width * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_out_values, h_values.data(),
            (size_t)total_count * sizeof(__half), cudaMemcpyHostToDevice));
    }

    ret.push_back(std::make_tuple(d_out_paths, d_out_values,
                                  total_count, path_width, effective_order));
    return ret;
}
