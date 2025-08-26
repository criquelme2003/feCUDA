#include "src/headers.cuh"
#include "src/utils.cuh"
#include "src/types.cuh"
#include <chrono>

extern "C"
{
    __global__ void max_min_kernel(
        const float *__restrict__ A,
        const float *__restrict__ B,
        float *__restrict__ C_min,
        float *__restrict__ C_max,
        const int M, const int K, const int N, const int batch_size);
}

void procesar_max_min(const char *archivo_A, const char *archivo_B,
                      const char *salida_min, const char *salida_max,
                      int batch, int M, int K, int N)
{

    printf("Procesando: %s + %s\n", archivo_A, archivo_B);

    // 1. Cargar datos (usa tus funciones existentes)
    TensorResult tensor_A, tensor_B;
    bool ok_A = leer_matriz_3d_desde_archivo(archivo_A, tensor_A, batch, M, K, 1);
    bool ok_B = leer_matriz_3d_desde_archivo(archivo_B, tensor_B, batch, K, N, 1);

    if (!ok_A || !ok_B)
    {
        printf("ERROR: No pudo cargar archivos\n");
        return;
    }

    // 2. Memoria GPU
    size_t size_A = batch * M * K * sizeof(float);
    size_t size_B = batch * K * N * sizeof(float);
    size_t size_min = batch * M * N * K * sizeof(float);
    size_t size_max = batch * M * N * sizeof(float);

    float *d_A, *d_B, *d_min, *d_max;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_min, size_min);
    cudaMalloc(&d_max, size_max);

    cudaMemcpy(d_A, tensor_A.data.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, tensor_B.data.data(), size_B, cudaMemcpyHostToDevice);

    // 3. Ejecutar kernel
    dim3 block(K);
    dim3 grid(N, M, batch);

    auto t1 = std::chrono::high_resolution_clock::now();
    max_min_kernel<<<grid, block, K * sizeof(float)>>>(d_A, d_B, d_min, d_max, M, K, N, batch);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    double tiempo_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 4. Copiar resultados
    std::vector<float> h_min(batch * M * N * K);
    std::vector<float> h_max(batch * M * N);

    cudaMemcpy(h_min.data(), d_min, size_min, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max.data(), d_max, size_max, cudaMemcpyDeviceToHost);

    // 5. Guardar archivos
    std::ofstream f_min(salida_min);
    for (float v : h_min)
        f_min << v << " ";
    f_min.close();

    std::ofstream f_max(salida_max);
    for (float v : h_max)
        f_max << v << " ";
    f_max.close();

    printf("  Tiempo: %.2f ms\n", tiempo_ms);
    printf("  Guardado: %s, %s\n\n", salida_min, salida_max);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_min);
    cudaFree(d_max);
}

int main()
{
    cuda_warmup();

    // Cambia estas rutas por las que quieras:
    procesar_max_min(
        "datasets_txt/CC.txt", "datasets_txt/CE.txt",
        "CC_CE_min.txt", "CC_CE_max.txt",
        10, 16, 16, 16);

    procesar_max_min(
        "datasets_txt/EE.txt", "datasets_txt/reflexive.txt",
        "EE_refl_min.txt", "EE_refl_max.txt",
        10, 16, 16, 16);

    return 0;
}
