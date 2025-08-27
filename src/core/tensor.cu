#include "core/tensor.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/logging.cuh"
#include <cuda_runtime.h>

namespace TensorUtils
{

    TensorResult create_device_tensor(int batch, int M, int N, int K)
    {
        const size_t size = batch * M * N * K * sizeof(float);
        float *data = nullptr;

        CHECK_CUDA(cudaMalloc(&data, size));
        return TensorResult(data, true, batch, M, N, K, true);
    }

    TensorResult create_host_tensor(int batch, int M, int N, int K)
    {
        const size_t size = batch * M * N * K * sizeof(float);
        float *data = static_cast<float *>(std::malloc(size));

        if (!data)
        {
            throw std::bad_alloc();
        }

        return TensorResult(data, false, batch, M, N, K, true);
    }

    TensorResult copy_to_device(const TensorResult &host_tensor)
    {
        if (host_tensor.is_device_ptr)
        {
            return host_tensor.clone(); // Ya está en device, solo clonar
        }

        TensorResult device_tensor = create_device_tensor(
            host_tensor.batch, host_tensor.M, host_tensor.N, host_tensor.K);

        const size_t size = host_tensor.size_bytes();
        CHECK_CUDA(cudaMemcpy(device_tensor.data, host_tensor.data, size, cudaMemcpyHostToDevice));

        return device_tensor;
    }

    TensorResult copy_to_host(const TensorResult &device_tensor)
    {
        if (!device_tensor.is_device_ptr)
        {
            return device_tensor.clone(); // Ya está en host, solo clonar
        }

        TensorResult host_tensor = create_host_tensor(
            device_tensor.batch, device_tensor.M, device_tensor.N, device_tensor.K);

        const size_t size = device_tensor.size_bytes();
        CHECK_CUDA(cudaMemcpy(host_tensor.data, device_tensor.data, size, cudaMemcpyDeviceToHost));

        return host_tensor;
    }

    bool are_compatible(const TensorResult &a, const TensorResult &b)
    {
        return a.batch == b.batch && a.M == b.M && a.N == b.N && a.K == b.K;
    }

    void fill_tensor(TensorResult &tensor, float value)
    {
        const size_t elements = tensor.total_elements();

        if (tensor.is_device_ptr)
        {
            // Usar cudaMemset para valores 0, o kernel personalizado para otros
            if (value == 0.0f)
            {
                CHECK_CUDA(cudaMemset(tensor.data, 0, tensor.size_bytes()));
            }
            else
            {
                // TODO: Implementar kernel para llenar con valor específico
                LOG_WARNING("fill_tensor con valor != 0 en device no implementado completamente");
            }
        }
        else
        {
            std::fill_n(tensor.data, elements, value);
        }
    }

    bool tensors_equal(const TensorResult &a, const TensorResult &b, float tolerance)
    {
        if (!are_compatible(a, b))
        {
            return false;
        }

        const size_t elements = a.total_elements();

        // Asegurar que ambos tensores están en host para comparación
        TensorResult host_a = a.is_device_ptr ? copy_to_host(a) : a;
        TensorResult host_b = b.is_device_ptr ? copy_to_host(b) : b;

        for (size_t i = 0; i < elements; ++i)
        {
            if (std::abs(host_a.data[i] - host_b.data[i]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }
}
