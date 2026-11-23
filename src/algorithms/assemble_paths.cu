#include "../../include/algorithms/assemble_paths.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

// Definición única de assemble_paths, compartida por maxmin v1/v2/v3.
// Ver documentación en include/algorithms/assemble_paths.cuh.
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
