#include "../../include/core/types.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <vector>

#define MAX_GRID_SIZE    10000
#define MAX_PATHS_PER_ITER 100000

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
    dim3 block(128);
    size_t shmem = 128 * (sizeof(__half) + sizeof(int));

    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN 1: un solo paso A⊗B
    // ─────────────────────────────────────────────────────────────────────────
    if (order == 1)
    {
        __half* C_out;
        CHECK_CUDA(cudaMalloc(&C_out, total_elems * sizeof(__half)));

        int*    d_raw_paths;
        __half* d_raw_values;
        int     h_count = 0;
        int*    d_counter;
        CHECK_CUDA(cudaMalloc(&d_counter,    sizeof(int)));
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_raw_paths,  (size_t)MAX_PATHS_PER_ITER * 4 * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_raw_values, (size_t)MAX_PATHS_PER_ITER * sizeof(__half)));

        if ((long long)B * M * N * K < MAX_PATHS_PER_ITER)
        {
            std::cout << "[MAXMIN C++] ORDER-1 COMPLETE" << std::endl;
            dim3 grid(N, M, B);
            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                d_A, d_B, C_out,
                d_raw_paths, d_raw_values, d_counter,
                /*argmax=*/nullptr,
                thr, B, M, N, K, -1);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
        }
        else
        {
            std::cout << "[MAXMIN C++] ORDER-1 BATCHED" << std::endl;
            int sizeA = M * K, sizeB = K * N;
            dim3 grid(N, M, 1);
            std::vector<int*>    d_paths_acc;
            std::vector<__half*> d_values_acc;
            std::vector<int>     h_counts_acc;

            int* d_ctr_b;
            CHECK_CUDA(cudaMalloc(&d_ctr_b, sizeof(int)));

            for (int b_ = 0; b_ < B; b_++)
            {
                int*    dp; __half* dv;
                int     mppi = M * K * N;
                CHECK_CUDA(cudaMalloc(&dp, (size_t)mppi * 4 * sizeof(int)));
                CHECK_CUDA(cudaMalloc(&dv, (size_t)mppi * sizeof(__half)));
                CHECK_CUDA(cudaMemset(d_ctr_b, 0, sizeof(int)));

                maxmin_threshold_kernel<<<grid, block, shmem>>>(
                    d_A + b_ * sizeA,
                    d_B + b_ * sizeB,
                    C_out + b_ * M * N,
                    dp, dv, d_ctr_b,
                    /*argmax=*/nullptr,
                    thr, 1, M, N, K, 0);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());

                int cnt = 0;
                CHECK_CUDA(cudaMemcpy(&cnt, d_ctr_b, sizeof(int), cudaMemcpyDeviceToHost));
                d_paths_acc.push_back(dp);
                d_values_acc.push_back(dv);
                h_counts_acc.push_back(cnt);
                h_count += cnt;
            }
            cudaFree(d_ctr_b);

            // Reusar d_raw_paths / d_raw_values para el merge
            cudaFree(d_raw_paths);
            cudaFree(d_raw_values);
            CHECK_CUDA(cudaMalloc(&d_raw_paths,  (size_t)h_count * 4 * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_raw_values, (size_t)h_count * sizeof(__half)));

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
                cudaFree(d_paths_acc[c]);
                cudaFree(d_values_acc[c]);
            }
        }
        cudaFree(C_out);
        cudaFree(d_counter);
        std::cout << "[MAXMIN C++] ORDER-1 paths found: " << h_count << std::endl;

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

            CHECK_CUDA(cudaMalloc(&d_final_paths,  (size_t)h_count * 5 * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_final_values, (size_t)h_count * sizeof(__half)));
            CHECK_CUDA(cudaMemcpy(d_final_paths, h_out.data(),
                (size_t)h_count * 5 * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_final_values, h_vals.data(),
                (size_t)h_count * sizeof(__half), cudaMemcpyHostToDevice));
        }
        cudaFree(d_raw_paths);
        cudaFree(d_raw_values);

        ret.push_back(std::make_tuple(d_final_paths, d_final_values,
                                      h_count, path_width, 1));
        return ret;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN > 1: iterativo con el mismo kernel
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "[MAXMIN C++] ORDER-" << order << " return_paths="
              << (return_paths ? "true" : "false") << std::endl;

    // Buffers iterativos: C_dev_before (= C_prev) y C_dev_after (= C_next)
    __half *C_dev_before, *C_dev_after;
    CHECK_CUDA(cudaMalloc(&C_dev_before, total_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&C_dev_after,  total_elems * sizeof(__half)));
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
        int*    d_paths_step;
        __half* d_values_step;
        int*    d_counter;
        CHECK_CUDA(cudaMalloc(&d_paths_step,  (size_t)MAX_PATHS_PER_ITER * 4 * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_values_step, (size_t)MAX_PATHS_PER_ITER * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_counter,     sizeof(int)));

        std::vector<int>    h_all_paths;  // acumula (order,b,m,k,n) de todos los steps
        std::vector<__half> h_all_values;

        for (int s = 0; s < order; s++)
        {
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                C_dev_before, d_B, C_dev_after,
                d_paths_step, d_values_step, d_counter,
                /*argmax=*/nullptr,
                thr, B, M, N, K, -1);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            int step_count = 0;
            CHECK_CUDA(cudaMemcpy(&step_count, d_counter, sizeof(int),
                                  cudaMemcpyDeviceToHost));

            if (step_count == 0) {
                std::cout << "[MAXMIN C++] Convergencia en step " << s + 1
                          << " (sin nuevas aristas)" << std::endl;
                break;
            }

            // Copiar paths del step a CPU y prepend (s+1)
            std::vector<int>    h_raw4((size_t)step_count * 4);
            std::vector<__half> h_vals((size_t)step_count);
            CHECK_CUDA(cudaMemcpy(h_raw4.data(), d_paths_step,
                (size_t)step_count * 4 * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_vals.data(), d_values_step,
                (size_t)step_count * sizeof(__half), cudaMemcpyDeviceToHost));

            prepend_order(h_raw4.data(), step_count, s + 1, h_all_paths);
            h_all_values.insert(h_all_values.end(), h_vals.begin(), h_vals.end());

            effective_order = s + 1;
            std::swap(C_dev_before, C_dev_after);
        }

        cudaFree(C_dev_before);
        cudaFree(C_dev_after);
        cudaFree(d_paths_step);
        cudaFree(d_values_step);
        cudaFree(d_counter);

        int total_count = (int)h_all_values.size();
        std::cout << "[MAXMIN C++] Total aristas (return_paths=false): "
                  << total_count << "  effective_order=" << effective_order << std::endl;

        int*   d_out_paths  = nullptr;
        __half* d_out_values = nullptr;
        if (total_count > 0) {
            CHECK_CUDA(cudaMalloc(&d_out_paths,  (size_t)total_count * 5 * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_out_values, (size_t)total_count * sizeof(__half)));
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
    // Copias CPU para detectar convergencia y reconstruir
    std::vector<__half> h_C_before(total_elems);  // C_{s} (antes del step actual)
    std::vector<__half> h_C_after (total_elems);  // C_{s+1} (resultado del kernel)
    // Guardamos las dos últimas para reconstrucción al final
    std::vector<__half> h_C_final    (total_elems);  // C_{effective_order}
    std::vector<__half> h_C_pre_final(total_elems);  // C_{effective_order - 1}

    CHECK_CUDA(cudaMemcpy(h_C_before.data(), d_A,
        total_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    std::vector<int*> d_argmax;
    d_argmax.reserve(order);

    for (int s = 0; s < order; s++)
    {
        int* d_am;
        CHECK_CUDA(cudaMalloc(&d_am, total_elems * sizeof(int)));
        d_argmax.push_back(d_am);

        maxmin_threshold_kernel<<<grid, block, shmem>>>(
            C_dev_before, d_B, C_dev_after,
            /*paths=*/nullptr, /*values=*/nullptr, /*counter=*/nullptr,
            d_am,
            thr, B, M, N, K, -1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C_after.data(), C_dev_after,
            total_elems * sizeof(__half), cudaMemcpyDeviceToHost));

        // Contar pares nuevos: C_after[i] - C_before[i] >= thr
        int new_count = 0;
        for (int i = 0; i < total_elems; i++)
            if (__half2float(h_C_after[i]) - __half2float(h_C_before[i]) >= thr_f)
                new_count++;

        if (new_count > 0) {
            effective_order = s + 1;
            h_C_pre_final = h_C_before;  // guardar C antes del último step
            h_C_final     = h_C_after;
            std::swap(h_C_before, h_C_after);     // h_C_before = C_{s+1} para el próximo
            std::swap(C_dev_before, C_dev_after);
        } else {
            std::cout << "[MAXMIN C++] Convergencia en step " << s + 1
                      << " (sin nuevos caminos)" << std::endl;
            cudaFree(d_argmax.back());
            d_argmax.pop_back();
            break;
        }
    }

    cudaFree(C_dev_before);
    cudaFree(C_dev_after);

    int n_steps = (int)d_argmax.size();  // = effective_order

    // Traer argmax a CPU
    std::vector<std::vector<int>> h_argmax(n_steps, std::vector<int>(total_elems));
    for (int s = 0; s < n_steps; s++) {
        CHECK_CUDA(cudaMemcpy(h_argmax[s].data(), d_argmax[s],
            total_elems * sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d_argmax[s]);
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
    std::cout << "[MAXMIN C++] Caminos reconstruidos (effective_order="
              << effective_order << "): " << total_count << std::endl;

    int*   d_out_paths  = nullptr;
    __half* d_out_values = nullptr;
    if (total_count > 0) {
        CHECK_CUDA(cudaMalloc(&d_out_paths,  (size_t)total_count * path_width * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_out_values, (size_t)total_count * sizeof(__half)));
        CHECK_CUDA(cudaMemcpy(d_out_paths, h_paths.data(),
            (size_t)total_count * path_width * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_out_values, h_values.data(),
            (size_t)total_count * sizeof(__half), cudaMemcpyHostToDevice));
    }

    ret.push_back(std::make_tuple(d_out_paths, d_out_values,
                                  total_count, path_width, effective_order));
    return ret;
}
