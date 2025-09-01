// EJEMPLO: Cómo usar las nuevas utilidades en maxmin.cu
// Este es un ejemplo de cómo refactorizar maxmin.cu usando las utilidades

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <maxmin_kernels.cuh>
#include <device_launch_parameters.h>
#include <core/types.cuh>
#include <headers.cuh>

// Incluir las nuevas utilidades
#include <utils/memory_utils.cuh>
#include <utils/validation_utils.cuh>

void maxmin_with_utils(const TensorResult &tensor1, const TensorResult &tensor2,
                       TensorResult &max_result, TensorResult &min_result,
                       bool keep_in_device)
{
    // 1. VALIDACIÓN usando ValidationUtils
    if (!ValidationUtils::InputValidator::validate_tensor_not_null(tensor1, "tensor1") ||
        !ValidationUtils::InputValidator::validate_tensor_not_null(tensor2, "tensor2"))
    {
        return;
    }

    // Validar dimensiones específicas del kernel
    if (tensor1.K != 1 || tensor2.K != 1)
    {
        std::cerr << "Error: maxmin_kernel_v1 solo acepta tensores 3D (K=1)\n";
        return;
    }

    // Validar dimensiones compatibles
    if (!ValidationUtils::InputValidator::validate_positive_dimensions(
            tensor1.M, tensor1.N, tensor1.K))
    {
        return;
    }

    // 2. EXTRAER DIMENSIONES
    int batch = tensor1.batch;
    int M = tensor1.M;
    int K = tensor1.N;
    int N = tensor2.N;

    printf("Ejecutando kernel_v1 con dimensiones: batch=%d, M=%d, K=%d, N=%d\n", batch, M, K, N);

    // 3. GESTIÓN DE MEMORIA usando MemoryUtils::CudaDevicePtr
    try
    {
        // Crear wrappers RAII para memoria CUDA - se liberan automáticamente
        MemoryUtils::CudaDevicePtr<float> d_A(batch * M * K);
        MemoryUtils::CudaDevicePtr<float> d_B(batch * K * N);
        MemoryUtils::CudaDevicePtr<float> d_C_min(batch * M * N * K);
        MemoryUtils::CudaDevicePtr<float> d_C_max(batch * M * N);

        // 4. COPIAR DATOS (si no están ya en device)
        if (!tensor1.is_device_ptr)
        {
            CHECK_CUDA(cudaMemcpy(d_A.get(), tensor1.data,
                                  batch * M * K * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        if (!tensor2.is_device_ptr)
        {
            CHECK_CUDA(cudaMemcpy(d_B.get(), tensor2.data,
                                  batch * K * N * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        // 5. CONFIGURAR Y LANZAR KERNEL
        dim3 block_size(16, 16);
        dim3 grid_size((N + block_size.x - 1) / block_size.x,
                       (M + block_size.y - 1) / block_size.y,
                       batch);

        // Llamar al kernel (ejemplo - ajustar según tu kernel real)
        // maxmin_kernel_v1<<<grid_size, block_size>>>(d_A.get(), d_B.get(),
        //                                             d_C_max.get(), d_C_min.get(),
        //                                             M, K, N, batch);

        CHECK_CUDA(cudaDeviceSynchronize());

        // 6. PREPARAR RESULTADOS usando HostPtr si necesario
        if (!keep_in_device)
        {
            // Alocar memoria host con RAII
            MemoryUtils::HostPtr<float> h_max_result(batch * M * N);
            MemoryUtils::HostPtr<float> h_min_result(batch * M * N * K);

            // Copiar resultados de device a host
            CHECK_CUDA(cudaMemcpy(h_max_result.get(), d_C_max.get(),
                                  batch * M * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            CHECK_CUDA(cudaMemcpy(h_min_result.get(), d_C_min.get(),
                                  batch * M * N * K * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Configurar TensorResult de salida (transferir ownership)
            max_result.data = h_max_result.release(); // transfiere ownership
            max_result.is_device_ptr = false;
            max_result.owns_memory = true;
            max_result.batch = batch;
            max_result.M = M;
            max_result.N = N;
            max_result.K = 1;

            min_result.data = h_min_result.release();
            min_result.is_device_ptr = false;
            min_result.owns_memory = true;
            min_result.batch = batch;
            min_result.M = M;
            min_result.N = N;
            min_result.K = K;
        }
        else
        {
            // Mantener en device - necesitarías crear copias permanentes
            // o usar otro patrón de gestión de memoria
        }

        // 7. La memoria CUDA se libera automáticamente cuando
        //    los CudaDevicePtr salen de scope
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error en maxmin_with_utils: " << e.what() << '\n';
        max_result = TensorResult();
        min_result = TensorResult();
    }
    // Nota: Toda la memoria se libera automáticamente gracias a RAII
}
