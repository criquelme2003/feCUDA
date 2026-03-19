#include "../../include/core/types.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

#define MAX_GRID_SIZE 10000
#define MAX_PATHS_PER_ITER 100000

std::vector<std::tuple<int*, __half*, int, int, int>>
maxmin(TensorResult<__half> &tensor1, TensorResult<__half> &tensor2, __half thr, int order)
{
    std::vector<std::tuple<int*, __half*, int, int, int>> ret;

    if (tensor1.getK() != 1 || tensor2.getK() != 1)
    {
        printf("Error: maxmin solo acepta tensores 3D (K=1)\n");
        exit(0); // tensor nulo
    }

    // Para el kernel, necesitamos que A sea [batch, M, K] y B sea [batch, K, N]
    // Pero como K=1, efectivamente son [batch, M] y [batch, N]
    int B = tensor1.getBatch();
    int M = tensor1.getM();
    int K = tensor1.getN(); // En el contexto del kernel, N del tensor1 es K
    int N = tensor2.getN();

    // Alocar memoria en device
    __half *d_A, *d_B;

    tensor1.move_to_device();
    tensor2.move_to_device();
    d_A = (__half *)tensor1.getData();
    d_B = (__half *)tensor2.getData();

    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN 1: kernel original con umbral (COMPLETE o BATCHED)
    // ─────────────────────────────────────────────────────────────────────────
    if (order == 1)
    {
        int4   *d_global_paths  = nullptr;
        __half *d_global_values = nullptr;
        int     h_total_count   = 0;

        if ((B * M * N * K) < MAX_PATHS_PER_ITER)
        {
            std::cout << "[MAXMIN C++] EXECUTING COMPLETE ALGORITHM" << std::endl;
            int *d_counter;
            dim3 block(128);
            dim3 grid(N, M, B);
            size_t shmem = 128 * sizeof(__half);
            CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_global_paths,  MAX_PATHS_PER_ITER * sizeof(int4)));
            CHECK_CUDA(cudaMalloc(&d_global_values, MAX_PATHS_PER_ITER * sizeof(__half)));

            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                d_A, d_B, d_global_paths, d_global_values,
                d_counter, thr, B, M, N, K, -1);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());

            CHECK_CUDA(cudaMemcpy(&h_total_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_counter);
        }
        else
        {
            std::cout << "[MAXMIN C++] EXECUTING BATCHED ALGORITHM" << std::endl;
            std::vector<__half *> d_values_acc;
            std::vector<int4  *> d_paths_acc;
            std::vector<int>     h_counter_acc;
            int mppi  = M * K * N;
            int sizeA = M * K;
            int sizeB = K * N;

            dim3 block(128);
            dim3 grid(N, M, 1);
            size_t shmem = 128 * sizeof(__half);

            int *d_counter;
            CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
            for (int b_ = 0; b_ < B; b_++)
            {
                int4   *d_paths;
                __half *d_values;
                CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
                CHECK_CUDA(cudaMalloc(&d_paths,  mppi * sizeof(int4)));
                CHECK_CUDA(cudaMalloc(&d_values, mppi * sizeof(__half)));

                maxmin_threshold_kernel<<<grid, block, shmem>>>(
                    d_A + b_ * sizeA, d_B + b_ * sizeB,
                    d_paths, d_values, d_counter, thr,
                    1, M, N, K, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaGetLastError());

                d_values_acc.push_back(d_values);
                d_paths_acc.push_back(d_paths);
                int temp = 0;
                CHECK_CUDA(cudaMemcpy(&temp, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
                h_total_count += temp;
                h_counter_acc.push_back(h_total_count);
            }
            cudaFree(d_counter);

            CHECK_CUDA(cudaMalloc(&d_global_paths,  h_total_count * sizeof(int4)));
            CHECK_CUDA(cudaMalloc(&d_global_values, h_total_count * sizeof(__half)));

            for (int c = 0; c < B; c++)
            {
                int offset     = (c > 0) ? h_counter_acc[c - 1] : 0;
                int count_here = h_counter_acc[c] - offset;
                CHECK_CUDA(cudaMemcpy(d_global_paths  + offset, d_paths_acc[c],
                    count_here * sizeof(int4),   cudaMemcpyDeviceToDevice));
                CHECK_CUDA(cudaMemcpy(d_global_values + offset, d_values_acc[c],
                    count_here * sizeof(__half), cudaMemcpyDeviceToDevice));
                cudaFree(d_paths_acc[c]);
                cudaFree(d_values_acc[c]);
            }
        }

        std::cout << "[MAXMIN C++] Paths found: " << h_total_count << std::endl;
        // cast int4* → int*: mismo layout en memoria (int4 = {x,y,z,w}, 4×int32)
        ret.push_back(std::make_tuple(
            reinterpret_cast<int*>(d_global_paths),
            d_global_values,
            h_total_count,
            4,  // path_width: (b, m, k, n)
            1   // effective_order
        ));
    }
    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN > 1: producto max-min iterativo + reconstrucción de caminos en CPU
    //
    //  Paso i:  C_{i+1}[b,m,n] = max_k  min(C_i[b,m,k], B[b,k,n])
    //           argmax[i][b,m,n] = k ganador
    //
    //  Reconstrucción hacia atrás desde (m,n):
    //    ks[order-1] = argmax[order-1][m, n]
    //    ks[order-2] = argmax[order-2][m, ks[order-1]]
    //    ...
    //    ks[0]       = argmax[0][m, ks[1]]
    //  Camino: m → ks[0] → ks[1] → ... → ks[order-1] → n
    // ─────────────────────────────────────────────────────────────────────────
    else
    {
        std::cout << "[MAXMIN C++] EXECUTING ORDER-" << order << " ALGORITHM" << std::endl;

        int total_elems = B * M * N;   // K == M == N para matrices cuadradas

        // ── 1. Buffers iterativos en device ──────────────────────────────────
        __half *C_prev, *C_next;
        CHECK_CUDA(cudaMalloc(&C_prev, total_elems * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&C_next, total_elems * sizeof(__half)));
        // C_prev arranca como copia de d_A
        CHECK_CUDA(cudaMemcpy(C_prev, d_A, total_elems * sizeof(__half), cudaMemcpyDeviceToDevice));

        // ── 2. Pre-copiar A_orig para comparar umbral en cada paso ───────────
        std::vector<__half> h_A_orig_pre(total_elems);
        CHECK_CUDA(cudaMemcpy(h_A_orig_pre.data(), d_A,
            total_elems * sizeof(__half), cudaMemcpyDeviceToHost));

        float thr_f_pre = __half2float(thr);
        auto count_passing = [&](const std::vector<__half>& C) -> int {
            int cnt = 0;
            for (int i = 0; i < total_elems; i++)
                if (__half2float(C[i]) - __half2float(h_A_orig_pre[i]) >= thr_f_pre)
                    cnt++;
            return cnt;
        };

        // ── 3. Argmax: un buffer por paso (se expande dinámicamente) ─────────
        std::vector<int*> d_argmax;
        d_argmax.reserve(order);

        dim3 block(128);
        dim3 grid(N, M, B);
        size_t shmem_step = 128 * (sizeof(__half) + sizeof(int));

        int effective_order = 0;  // último paso que aportó caminos nuevos
        int prev_pass_count = 0;

        for (int s = 0; s < order; s++)
        {
            int* d_am;
            CHECK_CUDA(cudaMalloc(&d_am, total_elems * sizeof(int)));
            d_argmax.push_back(d_am);

            maxmin_step_kernel<<<grid, block, shmem_step>>>(
                C_prev, d_B, C_next, d_argmax[s],
                B, M, N, K, -1);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            std::swap(C_prev, C_next);   // C_prev = C_{s+1}

            // Contar cuántos (m,n) superan el umbral tras este paso
            std::vector<__half> h_C_check(total_elems);
            CHECK_CUDA(cudaMemcpy(h_C_check.data(), C_prev,
                total_elems * sizeof(__half), cudaMemcpyDeviceToHost));

            int cur_pass = count_passing(h_C_check);
            if (cur_pass > prev_pass_count)
            {
                effective_order = s + 1;
                prev_pass_count = cur_pass;
            }
            else
            {
                // Este paso no aportó nada: liberar su argmax y detener
                cudaFree(d_argmax.back());
                d_argmax.pop_back();
                std::cout << "[MAXMIN C++] Convergencia en orden " << s + 1
                          << " (sin nuevos caminos, deteniendo)" << std::endl;
                break;
            }
        }
        // d_argmax tiene exactamente effective_order entradas
        // C_prev = producto final tras effective_order pasos
        cudaFree(C_next);

        // ── 4. Traer a CPU: C_final y todos los argmax ───────────────────────
        int n_steps = (int)d_argmax.size();   // = effective_order

        std::vector<__half> h_C_final(total_elems);
        std::vector<std::vector<int>> h_argmax(n_steps, std::vector<int>(total_elems));

        CHECK_CUDA(cudaMemcpy(h_C_final.data(), C_prev,
            total_elems * sizeof(__half), cudaMemcpyDeviceToHost));
        for (int s = 0; s < n_steps; s++)
        {
            CHECK_CUDA(cudaMemcpy(h_argmax[s].data(), d_argmax[s],
                total_elems * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_argmax[s]);
        }
        cudaFree(C_prev);

        // ── 5. Reconstrucción de caminos en CPU ──────────────────────────────
        int path_width = n_steps + 3;   // (b, m, k1, …, k_{n_steps}, n)
        std::vector<int>    h_paths;
        std::vector<__half> h_values;
        float thr_f = __half2float(thr);

        for (int b_ = 0; b_ < B; b_++)
        {
            for (int m_ = 0; m_ < M; m_++)
            {
                for (int n_ = 0; n_ < N; n_++)
                {
                    int idx    = b_ * M * N + m_ * N + n_;
                    float val  = __half2float(h_C_final[idx]);
                    float orig = __half2float(h_A_orig_pre[idx]);

                    if ((val - orig) >= thr_f)
                    {
                        std::vector<int> ks(n_steps);
                        int cur = n_;
                        for (int s = n_steps - 1; s >= 0; s--)
                        {
                            ks[s] = h_argmax[s][b_ * M * N + m_ * N + cur];
                            cur   = ks[s];
                        }

                        h_paths.push_back(b_);
                        h_paths.push_back(m_);
                        for (int k : ks) h_paths.push_back(k);
                        h_paths.push_back(n_);
                        h_values.push_back(h_C_final[idx]);
                    }
                }
            }
        }

        int h_total_count = (int)h_values.size();
        std::cout << "[MAXMIN C++] Paths found (effective_order=" << effective_order
                  << " of max=" << order << "): " << h_total_count << std::endl;

        // ── 5. Subir resultados a device ─────────────────────────────────────
        int   *d_flat_paths = nullptr;
        __half *d_vals      = nullptr;

        if (h_total_count > 0)
        {
            CHECK_CUDA(cudaMalloc(&d_flat_paths, (size_t)h_total_count * path_width * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_vals,       (size_t)h_total_count * sizeof(__half)));
            CHECK_CUDA(cudaMemcpy(d_flat_paths, h_paths.data(),
                (size_t)h_total_count * path_width * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_vals, h_values.data(),
                (size_t)h_total_count * sizeof(__half), cudaMemcpyHostToDevice));
        }

        ret.push_back(std::make_tuple(d_flat_paths, d_vals, h_total_count, path_width, effective_order));
    }

    return ret;
}
