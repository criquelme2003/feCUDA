#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "types.cuh"

// Macro para verificar errores de CUDA
#define CHECK_CUDA(call)                                          \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

// Macro para verificar errores de CUDA sin salir del programa
#define CHECK_CUDA_SAFE(call)                                     \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false;                                         \
        }                                                         \
    }

// Función para imprimir TensorResult
void imprimir_tensor(const TensorResult &tensor, int max_rows = 10, int max_cols = 10,
                     const char *nombre_tensor = "Tensor", bool mostrar_estadisticas = true);

// Función para leer matrices 3D desde archivo de texto
bool leer_matriz_3d_desde_archivo(const char *archivo, TensorResult &tensor,
                                  int batch, int M, int N, int K = 1);

// Función para guardar tensor como archivo en formato compatible con leer_matriz_3d_desde_archivo
bool guardar_tensor_como_archivo(const TensorResult &tensor, const char *nombre_archivo);

// Función para comparar dos tensores elemento a elemento
bool comparar_tensores(const TensorResult &tensor_a, const TensorResult &tensor_b,
                       float tolerancia = 1e-6f, bool verbose = true);

// Tipo de función para algoritmos maxmin
typedef TensorResult (*FuncionMaxMin)(const TensorResult &tensor_a, const TensorResult &tensor_b);

// Función para validar algoritmos maxmin contra archivos de referencia
void validar_algoritmos_maxmin(FuncionMaxMin funcion_maxmin, const char *nombre_algoritmo = "MaxMin");

// Función para limpiar y verificar el estado del dispositivo CUDA
void cuda_cleanup_and_check();

// Función para limpiar memoria de TensorResult de forma segura
void safe_tensor_cleanup(TensorResult &tensor);

// Función para crear una copia del tensor en memoria host
TensorResult copy_tensor(const TensorResult &src);

// Función para calentar el sistema CUDA
void cuda_warmup();

// Función transpose
__global__ void transpose_kernel_optimized(
    const float *__restrict__ input, // [batch, K, N]
    float *__restrict__ output,      // [batch, N, K]
    int K, int N, int batch_size);

#endif
