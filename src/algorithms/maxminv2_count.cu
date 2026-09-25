#include "../../include/algorithms/assemble_paths.cuh"
#include "../../include/core/types.cuh"
#include "../../include/headers.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/kernels/path_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cub/cub.cuh>
#include <sys/stat.h>
#include <string>
// g_verbose se define en maxmin.cu (variante v1); aquí sólo se usa vía extern.

#define MAX_GRID_SIZE 10000
#define MAX_PATHS_PER_ITER 100000

// ─────────────────────────────────────────────────────────────────────────────
// Persistencia a disco de los CSR de testigos, para reconstrucción de
// caminos on-demand (ver diseño acordado: no se expande a caminos completos
// aquí, sólo se guarda la topología + valores por nivel).
//
// Por orden s con efectos: 3 binarios crudos (C, row_offsets, csr_pivots) +
// una entrada en manifest.json con las dimensiones necesarias para leerlos
// después (sin eso, los .bin crudos no se pueden interpretar).
// ─────────────────────────────────────────────────────────────────────────────

// Escribe un buffer host a un archivo binario crudo, sin metadata ni
// interpretación de formato: solo el dump de bytes.
static void write_bin(const std::string &path, const void *data, size_t bytes) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "ERROR: no se pudo abrir " << path << " para escritura" << std::endl;
        return;
    }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

// Vuelca un nivel completo (C, row_offsets, csr_pivots) a dump_dir y agrega
// su entrada al manifiesto abierto (manifest_f), en formato JSON, un objeto
// por línea dentro del array "orders".
static void dump_order_level(
    const std::string &dump_dir,
    FILE *manifest_f,
    bool &manifest_first_entry,
    int s,
    int M,
    int N,
    int K,
    int Mpad,
    int Npad,
    size_t padded_elems,
    int h_counter,
    float thr_f,
    const __half *d_C,
    const int *d_row_offsets,
    const int *d_csr_pivots
) {
    std::vector<__half> h_C(padded_elems);
    std::vector<int> h_row_offsets(padded_elems);
    std::vector<int> h_csr_pivots(h_counter);

    CHECK_CUDA(cudaMemcpy(
        h_C.data(), d_C, padded_elems * sizeof(__half), cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaMemcpy(
        h_row_offsets.data(), d_row_offsets, padded_elems * sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaMemcpy(
        h_csr_pivots.data(), d_csr_pivots, h_counter * sizeof(int), cudaMemcpyDeviceToHost
    ));

    std::string file_C           = "order_" + std::to_string(s) + "_C.bin";
    std::string file_row_offsets = "order_" + std::to_string(s) + "_row_offsets.bin";
    std::string file_csr_pivots  = "order_" + std::to_string(s) + "_csr_pivots.bin";

    write_bin(dump_dir + "/" + file_C, h_C.data(), padded_elems * sizeof(__half));
    write_bin(
        dump_dir + "/" + file_row_offsets, h_row_offsets.data(), padded_elems * sizeof(int)
    );
    write_bin(
        dump_dir + "/" + file_csr_pivots, h_csr_pivots.data(), h_counter * sizeof(int)
    );

    if (!manifest_first_entry) fprintf(manifest_f, ",\n");
    manifest_first_entry = false;
    fprintf(
        manifest_f,
        "    {\n"
        "      \"s\": %d,\n"
        "      \"M\": %d, \"N\": %d, \"K\": %d,\n"
        "      \"Mpad\": %d, \"Npad\": %d,\n"
        "      \"padded_elems\": %zu,\n"
        "      \"h_counter\": %d,\n"
        "      \"thr\": %.6f,\n"
        "      \"files\": {\n"
        "        \"C\": \"%s\",\n"
        "        \"row_offsets\": \"%s\",\n"
        "        \"csr_pivots\": \"%s\"\n"
        "      }\n"
        "    }",
        s, M, N, K, Mpad, Npad, padded_elems, h_counter, thr_f,
        file_C.c_str(), file_row_offsets.c_str(), file_csr_pivots.c_str()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Prepend step_order a cada fila (b,m,k,n) y acumula en out_host.
// raw4: buffer host con count*4 ints   layout: [b, m, k, n] por fila
// out_host: vector acumulador con path_width=5
// ─────────────────────────────────────────────────────────────────────────────
// Ensambla o extiende caminos usando el total_pivots que ya calculó el kernel en GPU.
//
// Primera llamada (prev_paths vacío):
//   Para cada (b,m,n) donde C[m,n] - A[m,n] >= thr, el pivote k viene
//   directamente de total_pivots[b,m,n]. Devuelve paths [b, m, k, n].
//
// Llamadas siguientes:
//   Para cada path [b,m,...,n], busca n2 donde C[m,n2] - A[m,n2] >= thr
//   y total_pivots[b,m,n2] == n (el kernel eligió n como pivote óptimo).
//   Devuelve paths extendidos con n2 al final.
//
MaxminResult maxminv2_count(
    TensorResult<__half> &tensor1,
    TensorResult<__half> &tensor2,
    __half thr,
    int order,
    const char *dump_dir
) {
    // Persistencia a disco (opcional): crea dump_dir si no existe y abre
    // manifest.json para ir acumulando una entrada por orden con efectos.
    FILE *manifest_f = nullptr;
    bool manifest_first_entry = true;
    std::string dump_dir_str;
    if (dump_dir != nullptr) {
        dump_dir_str = dump_dir;
        mkdir(dump_dir_str.c_str(), 0755); // no falla si ya existe; ignoramos el resultado
        std::string manifest_path = dump_dir_str + "/manifest.json";
        manifest_f = fopen(manifest_path.c_str(), "w");
        if (!manifest_f) {
            std::cerr << "ERROR: no se pudo abrir " << manifest_path << std::endl;
        } else {
            fprintf(manifest_f, "{\n  \"orders\": [\n");
        }
    }

    if (tensor1.getK() != 1 || tensor2.getK() != 1) {
        printf("Error: maxminv2 solo acepta tensores 3D (K=1)\n");
        exit(0);
    }

    // GET NECESARY DIMENSIONS (extents lógicos)
    int B = tensor1.getBatch();
    int M = tensor1.getM();
    int K = tensor1.getN(); // N del tensor1 actúa como K en el kernel
    int N = tensor2.getN();

    // Padea a múltiplo de 32 (tamaño de tile del kernel v2). El kernel indexa
    // con strides físicos.
    tensor1.move_to_device(32);
    tensor2.move_to_device(32);
    __half *d_A = (__half *)tensor1.getData();
    __half *d_B = (__half *)tensor2.getData();

    // Dimensiones FÍSICAS (padded). En este flujo K==N ⇒ Kpad==Npad, pero se
    // mantienen separadas por corrección general.
    int Mpad = tensor1.getMPadded();
    int Kpad = tensor1.getNPadded(); // N lógico de tensor1 = K → su padding es Kpad
    int Npad = tensor2.getNPadded();

    // Buffers internos con layout físico [B, Mpad, Npad] (== [B,Mpad,Kpad] aquí).
    size_t padded_elems = (size_t)B * Mpad * Npad;

    // Configuración de lanzamiento (fija para todos los paths)
    int blockDim = BLOCKSIZE * BLOCKSIZE;
    dim3 block(blockDim); // define steps for thread loop

    // ─────────────────────────────────────────────────────────────────────────
    // ORDEN > 1: iterativo con el mismo kernel
    // ─────────────────────────────────────────────────────────────────────────

    LOG(std::cout << "[MAXMIN C++] M: " << M << "B: " << B << std::endl);

    LOG(std::cout << "[MAXMIN C++] ORDER-" << order << std::endl);

    // Buffers iterativos con layout físico padded: C_dev_before (= C_prev) y
    // C_dev_after (= C_next). Se rellenan con negativo (0xBC) para que el padding
    // no gane el max ni contamine el counter.
    __half *C_dev_before, *C_dev_after;
    int *total_pivots;
    {
        size_t __alloc_bytes = padded_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_before");
        CHECK_CUDA(cudaMalloc(&C_dev_before, __alloc_bytes));
        CHECK_CUDA(cudaMemset(C_dev_before, 0xBC, __alloc_bytes));
    }
    {
        size_t __alloc_bytes = padded_elems * sizeof(__half);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "C_dev_after");
        CHECK_CUDA(cudaMalloc(&C_dev_after, __alloc_bytes));
        CHECK_CUDA(cudaMemset(C_dev_after, 0xBC, __alloc_bytes));
    }

    {
        size_t __alloc_bytes = padded_elems * sizeof(int);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "total_pivots");
        CHECK_CUDA(cudaMalloc(&total_pivots, __alloc_bytes));
        CHECK_CUDA(cudaMemset(total_pivots, 0, __alloc_bytes));
    }
    // C_dev_before arranca como copia física de d_A (ambos [B,Mpad,Kpad]==[B,Mpad,Npad]).
    CHECK_CUDA(
        cudaMemcpy(C_dev_before, d_A, padded_elems * sizeof(__half), cudaMemcpyDeviceToDevice)
    );

    // Número de tiles 32×32 (usar extent lógico; los tiles de borde cubren padding).
    dim3 grid(CEIL_DIV(Mpad, 32), CEIL_DIV(Npad, 32), B);
    float thr_f = __half2float(thr);
    int effective_order = 1;

    unsigned long long *d_counter;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(unsigned long long)));

    // row_offsets del CSR de testigos: offset exclusivo acumulado de
    // total_pivots, uno por celda (row_offsets[i] = suma de pivots[0..i-1]).
    // El total de testigos NO sale de aquí — ya se conoce vía h_counter, el
    // resultado del kernel de conteo. Vive en device; se recalcula cada
    // orden porque total_pivots cambia.
    int *d_row_offsets;
    {
        size_t __alloc_bytes = padded_elems * sizeof(int);
        CHECK_ALLOC_SIZE_OR_EXIT(__alloc_bytes, "row_offsets");
        CHECK_CUDA(cudaMalloc(&d_row_offsets, __alloc_bytes));
    }

    // Scratch de cub::DeviceScan — tamaño fijo para padded_elems, se
    // dimensiona una vez (dry-run) y se reutiliza en todas las iteraciones.
    void *d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    CHECK_CUDA(
        cub::DeviceScan::ExclusiveSum(
            d_scan_temp,
            scan_temp_bytes,
            total_pivots,
            d_row_offsets,
            (int)padded_elems
        )
    );
    CHECK_CUDA(cudaMalloc(&d_scan_temp, scan_temp_bytes));

    std::vector<std::vector<int>> current_paths;
    MaxminResult result;

    for (int s = 0; s < order; s++) {
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(unsigned long long)));
        // total_pivots sólo se escribe (en el kernel) para celdas que pasan
        // thr en ESTA iteración; sin resetear aquí, las celdas que no pasan
        // retienen el conteo de una iteración anterior, contaminando el
        // scan de row_offsets con offsets que no corresponden al csr_pivots
        // real de esta iteración (mucho más chico, sólo h_counter elementos).
        CHECK_CUDA(cudaMemset(total_pivots, 0, padded_elems * sizeof(int)));

        maxmin_all_pivots<<<grid, block>>>(
            C_dev_before,
            d_B,
            C_dev_after,
            total_pivots,
            d_counter,
            thr,
            B,
            M,
            N,
            K,
            Kpad,
            Npad,
            -1
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        int h_counter = 0;
        CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_counter == 0) {
            LOG(std::cout << "[MAXMIN C++] Convergencia en step " << s + 1 << std::endl);
            break;
        } else {
            LOG(std::cout << "[MAXMIN C++] Efectos encontrados en orden " << s + 1 << " : "
                          << h_counter << std::endl);
        }

        // Offsets del CSR de testigos: scan exclusivo de total_pivots en GPU
        // (evita el roundtrip D2H de hasta padded_elems ints que hacía la
        // versión en host). row_offsets[i] = suma de pivots[0..i-1]; el
        // total (== h_counter) es row_offsets[padded_elems-1] + pivots[padded_elems-1].
        CHECK_CUDA(
            cub::DeviceScan::ExclusiveSum(
                d_scan_temp,
                scan_temp_bytes,
                total_pivots,
                d_row_offsets,
                (int)padded_elems
            )
        );

        // Registrar los efectos de este orden (1 entrada por orden con efectos).
        int *csr_pivots;
        cudaMalloc(&csr_pivots, h_counter * sizeof(int));
        effective_order = s + 1;
        maxmin_write_all_pivots<<<grid, block>>>(
            C_dev_before,
            d_B,
            C_dev_after,
            total_pivots,
            d_row_offsets,
            csr_pivots,
            B,
            M,
            N,
            K,
            Kpad,
            Npad,
            -1
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Persistencia a disco de este nivel (si se pidió dump_dir). C_dev_after
        // trae el max-min recién calculado en esta iteración — es el C[s] a
        // guardar (antes del swap, que lo convierte en C_dev_before del próximo s).
        if (manifest_f != nullptr) {
            dump_order_level(
                dump_dir_str, manifest_f, manifest_first_entry, s, M, N, K, Mpad, Npad,
                padded_elems, h_counter, thr_f, C_dev_after, d_row_offsets, csr_pivots
            );
        }

        CHECK_CUDA(cudaFree(csr_pivots));

        result.effects_per_order.push_back(h_counter);

        std::swap(C_dev_before, C_dev_after);
    }

    if (manifest_f != nullptr) {
        fprintf(manifest_f, "\n  ]\n}\n");
        fclose(manifest_f);
    }

    CHECK_CUDA(cudaFree(d_counter));
    CHECK_CUDA(cudaFree(C_dev_before));
    CHECK_CUDA(cudaFree(C_dev_after));
    CHECK_CUDA(cudaFree(total_pivots));
    CHECK_CUDA(cudaFree(d_row_offsets));
    CHECK_CUDA(cudaFree(d_scan_temp));

    result.effective_order = effective_order;
    return result;
}
