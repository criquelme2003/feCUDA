#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <types.cuh>

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

// Función para limpiar y verificar el estado del dispositivo CUDA
void cuda_cleanup_and_check();

// Función para limpiar memoria de TensorResult de forma segura
void safe_tensor_cleanup(TensorResult &tensor);

// Función para crear una copia del tensor en memoria host
TensorResult copy_tensor(const TensorResult &src);

#endif
