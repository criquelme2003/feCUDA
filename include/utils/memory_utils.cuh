#ifndef MEMORY_UTILS_CUH
#define MEMORY_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// Incluir la macro CHECK_CUDA
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess)                                                 \
        {                                                                       \
            std::string error_msg = std::string("CUDA error at ") +             \
                                    __FILE__ + ":" + std::to_string(__LINE__) + \
                                    ": " + cudaGetErrorString(err);             \
            std::cerr << error_msg << std::endl;                                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }
#endif

namespace MemoryUtils
{

    // Gestor de memoria CUDA simple con RAII
    struct CudaMemoryManager
    {
        static void *allocate_device(size_t size)
        {
            void *ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&ptr, size));
            return ptr;
        }

        static void deallocate_device(void *ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
            }
        }

        static void *allocate_host(size_t size)
        {
            void *ptr = std::malloc(size);
            if (!ptr)
            {
                throw std::bad_alloc();
            }
            return ptr;
        }

        static void deallocate_host(void *ptr)
        {
            if (ptr)
            {
                std::free(ptr);
            }
        }
    };

    // Wrapper RAII para memoria CUDA
    template <typename T>
    struct CudaDevicePtr
    {
        T *ptr;
        bool owns_memory;

        explicit CudaDevicePtr(size_t count) : owns_memory(true)
        {
            ptr = static_cast<T *>(CudaMemoryManager::allocate_device(count * sizeof(T)));
        }

        // Constructor para punteros existentes (no toma ownership)
        explicit CudaDevicePtr(T *existing_ptr) : ptr(existing_ptr), owns_memory(false) {}

        ~CudaDevicePtr()
        {
            if (owns_memory)
            {
                CudaMemoryManager::deallocate_device(ptr);
            }
        }

        // No copiable, solo movible
        CudaDevicePtr(const CudaDevicePtr &) = delete;
        CudaDevicePtr &operator=(const CudaDevicePtr &) = delete;

        CudaDevicePtr(CudaDevicePtr &&other) noexcept
            : ptr(other.ptr), owns_memory(other.owns_memory)
        {
            other.ptr = nullptr;
            other.owns_memory = false;
        }

        T *get() const { return ptr; }
        operator T *() const { return ptr; }
    };

    // Wrapper RAII para memoria host
    template <typename T>
    struct HostPtr
    {
        T *ptr;

        explicit HostPtr(size_t count)
        {
            ptr = static_cast<T *>(CudaMemoryManager::allocate_host(count * sizeof(T)));
        }

        ~HostPtr()
        {
            CudaMemoryManager::deallocate_host(ptr);
        }

        // No copiable, solo movible
        HostPtr(const HostPtr &) = delete;
        HostPtr &operator=(const HostPtr &) = delete;

        HostPtr(HostPtr &&other) noexcept : ptr(other.ptr)
        {
            other.ptr = nullptr;
        }

        T *get() const { return ptr; }
        operator T *() const { return ptr; }
        T *release()
        {
            T *temp = ptr;
            ptr = nullptr;
            return temp;
        }
    };

} // namespace MemoryUtils

#endif // MEMORY_UTILS_CUH
