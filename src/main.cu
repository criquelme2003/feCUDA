#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <tuple>
#include "headers.cuh"
#include "utils.cuh"
#include "types.cuh"
#include <chrono>
#include <cublas_v2.h>

// Declarar la función wrapper del kernel_v1
extern TensorResult maxmin_kernel_v1_wrapper(const TensorResult &tensor_a, const TensorResult &tensor_b);

// Declaraciones de funciones
void ejecutar_benchmark_original();
void testing_traspose();
void validar_kernel_v1();
void ejecutar_max_min_con_archivos(const char *archivo_A, const char *archivo_B,
                                   const char *output_min, const char *output_max,
                                   int batch_size, int M, int K, int N);
int main()
{
    printf("=== SISTEMA DE PRUEBAS FECUDA ===\n");

    int opcion;
    printf("1. Ejecutar max_min con archivos\n");
    printf("2. Validar Kernel V1\n");
    printf("3. Benchmark original\n");
    printf("Selecciona opción: ");
    scanf("%d", &opcion);

    switch (opcion)
    {
    case 1:
        // Ejemplo: reflexive vs reflexive
        ejecutar_max_min_con_archivos(
            "datasets_txt/reflexive.txt",
            "datasets_txt/reflexive.txt",
            "results/reflexive_min.txt",
            "results/reflexive_max.txt",
            1, 6, 6, 6);
        break;
    case 2:
        validar_kernel_v1();
        break;
    case 3:
        ejecutar_benchmark_original();
        break;
    default:
        printf("Opción inválida\n");
    }
}

void testing_traspose()
{

    int batch_size = 2, K = 2, N = 3;

    int size = batch_size * K * N;
    // Host: batch de 2 matrices [2,2,3]

    float *h_input = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        h_input[i] = i + 1;
    }

    float *h_output = (float *)malloc(size * sizeof(float));

    // Device
    float *d_input;
    float *d_output;
    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));

    cudaMemcpy(d_input, h_input,
               size * sizeof(float),
               cudaMemcpyHostToDevice);

    // Handle cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Transposición
    transpose_kernel_optimized<<<dim3((N + 31) / 32, (K + 31) / 32, batch_size), dim3(32, 32)>>>(
        d_input, d_output, K, N, batch_size);

    cudaDeviceSynchronize();
    // Copiar de vuelta
    cudaMemcpy(h_output, d_output,
               size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Ver resultados
    std::cout << "Output:" << std::endl;
    for (int b = 0; b < batch_size; b++)
    {
        std::cout << "Matriz " << b << " (transpuesta):" << std::endl;
        for (int n = 0; n < N; n++)
        {
            for (int k = 0; k < K; k++)
            {
                std::cout << h_output[b * N * K + n * K + k] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cublasDestroy(handle);
}

void ejecutar_benchmark_original()
{
    cuda_warmup();

    printf("\n=== BENCHMARK ORIGINAL (iterative_maxmin_cuadrado) ===\n");

    TensorResult tensor_desde_archivo;
    bool exito = leer_matriz_3d_desde_archivo("../datasets_txt/CC.txt", tensor_desde_archivo, 10, 16, 16, 1);

    if (!exito)
    {
        printf("\nError: No se pudo cargar el tensor desde archivo\n");
    }

    // === Probando función iterative_maxmin_cuadrado ===

    // Usar el tensor cargado desde archivo si está disponible, sino usar datos hardcodeados
    printf("\n=== Probando función iterative_maxmin_cuadrado ===\n");
    TensorResult test_tensor;
    bool usar_archivo = exito;
    test_tensor = tensor_desde_archivo;

    std::vector<TensorResult> result_tensor_paths;
    std::vector<TensorResult> result_values_paths;
    std::vector<TensorResult> pure_tensor_paths;
    std::vector<TensorResult> pure_values_paths;

    float test_threshold = 0.4;
    int test_order = 4;

    iterative_maxmin_cuadrado(test_tensor, test_threshold, test_order, result_tensor_paths, result_values_paths, pure_tensor_paths, pure_values_paths);
}

void test_max_min_simple()
{
    printf("\n=== TEST MAX_MIN KERNEL SIMPLE ===\n");

    cuda_warmup();

    int batch_size = 1, M = 2, K = 3, N = 2;

    // Datos de prueba A[1,2,3,4,5,6] -> matrices [1,2,3] y [4,5,6]
    // Datos de prueba B[1,2,3,4,5,6] -> matrices [1,2], [3,4], [5,6]
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {1, 2, 3, 4, 5, 6};

    size_t size_A = batch_size * M * K * sizeof(float);
    size_t size_B = batch_size * K * N * sizeof(float);
    size_t size_C_min = batch_size * M * N * K * sizeof(float);
    size_t size_C_max = batch_size * M * N * sizeof(float);

    float *d_A, *d_B, *d_C_min, *d_C_max;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_min, size_C_min);
    cudaMalloc(&d_C_max, size_C_max);

    cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice);

    dim3 blockSize(K);
    dim3 gridSize(N, M, batch_size);
    size_t shared_mem = K * sizeof(float);

    auto inicio = std::chrono::high_resolution_clock::now();

    max_min_kernel<<<gridSize, blockSize, shared_mem>>>(
        d_A, d_B, d_C_min, d_C_max, M, K, N, batch_size);

    cudaDeviceSynchronize();
    auto fin = std::chrono::high_resolution_clock::now();

    double tiempo = std::chrono::duration<double, std::milli>(fin - inicio).count();

    std::vector<float> h_C_min(batch_size * M * N * K);
    std::vector<float> h_C_max(batch_size * M * N);

    cudaMemcpy(h_C_min.data(), d_C_min, size_C_min, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_max.data(), d_C_max, size_C_max, cudaMemcpyDeviceToHost);

    printf("Tiempo ejecución: %.3f ms\n", tiempo);
    printf("C_min (%d elementos): ", (int)h_C_min.size());
    for (int i = 0; i < (int)h_C_min.size(); i++)
    {
        printf("%.1f ", h_C_min[i]);
    }
    printf("\nC_max (%d elementos): ", (int)h_C_max.size());
    for (int i = 0; i < (int)h_C_max.size(); i++)
    {
        printf("%.1f ", h_C_max[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_min);
    cudaFree(d_C_max);
}

void ejecutar_max_min_con_archivos(const char *archivo_A, const char *archivo_B,
                                   const char *output_min, const char *output_max,
                                   int batch_size, int M, int K, int N)
{

    printf("Procesando: %s y %s\n", archivo_A, archivo_B);

    // Cargar datos
    TensorResult tensor_A, tensor_B;
    bool exito_A = leer_matriz_3d_desde_archivo(archivo_A, tensor_A, batch_size, M, K, 1);
    bool exito_B = leer_matriz_3d_desde_archivo(archivo_B, tensor_B, batch_size, K, N, 1);

    if (!exito_A || !exito_B)
    {
        printf("Error cargando archivos\n");
        return;
    }

    // Tamaños
    size_t size_A = batch_size * M * K * sizeof(float);
    size_t size_B = batch_size * K * N * sizeof(float);
    size_t size_C_min = batch_size * M * N * K * sizeof(float);
    size_t size_C_max = batch_size * M * N * sizeof(float);

    // GPU memory
    float *d_A, *d_B, *d_C_min, *d_C_max;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_min, size_C_min);
    cudaMalloc(&d_C_max, size_C_max);

    // Copiar a GPU
    cudaMemcpy(d_A, tensor_A.data, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, tensor_B.data, size_B, cudaMemcpyHostToDevice);

    // Configurar kernel
    dim3 blockSize(K);
    dim3 gridSize(N, M, batch_size);
    size_t shared_mem = K * sizeof(float);

    // Ejecutar con timing

    max_min_kernel<<<gridSize, blockSize, shared_mem>>>(
        d_A, d_B, d_C_min, d_C_max, M, K, N, batch_size);

    cudaDeviceSynchronize();

    // Copiar resultados
    float *h_C_min = new float[batch_size * M * N * K];
    float *h_C_max = new float[batch_size * M * N];

    cudaMemcpy(h_C_min, d_C_min, size_C_min, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_max, d_C_max, size_C_max, cudaMemcpyDeviceToHost);

    TensorResult max(h_C_min, false, batch_size, M, N, 1);

    guardar_tensor_como_archivo(max, output_min);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_min);
    cudaFree(d_C_max);
}

void validar_kernel_v1()
{
    printf("\n=== VALIDACIÓN KERNEL_V1 ===\n");

    cuda_warmup();

    // Usar la función de validación automática
    validar_algoritmos_maxmin(maxmin_kernel_v1_wrapper, "Kernel_V1_GPU");
}
