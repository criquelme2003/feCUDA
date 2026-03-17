#ifndef TYPES_CUH
#define TYPES_CUH
#include "../../include/utils.cuh"
#include "dlpack/dlpack.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <limits.h>
#include <limits>
#include <vector>

#include <pybind11/pybind11.h>
namespace py = pybind11;

enum class MemorySpace
{
    Host,
    Device
};

struct TensorResultDims
{
    int b = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    int getTotal() const
    {
        return b * m * n * k;
    }
};

inline int64_t *make_c_strides(const std::vector<int64_t> &shape)
{
    int ndim = shape.size();
    int64_t *strides = static_cast<int64_t *>(std::malloc(ndim * sizeof(int64_t)));

    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i)
    {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

// Estructura para mantener información completa del tensor
template <typename T = float> struct TensorResult
{
  private:
    T *data;               // Puntero a los datos
    TensorResultDims dims; // Dimensiones del tensor (K para dimensiones adicionales)
    MemorySpace space;
    bool released = false;

  public:
    DLManagedTensor *managed = nullptr;
    // Constructor completo con ownership
    TensorResult(MemorySpace memory_space, int b, int m, int n, int k = 1, bool owns = true)
        : data(nullptr), space(memory_space), dims({b, m, n, k})
    {
        allocateData();
    }

    // Reservar memoria para el tensor
    void allocateData()
    {

        // Controlar dimensiones
        std::vector<int> ds = {dims.b, dims.m, dims.n, dims.k};

        for (int ix = 0; ix < ds.size(); ix++)
        {
            std::vector<int> ds_copy(ds);

            ds_copy.erase(ds_copy.begin() + ix);

            int mult = 1;
            for (auto el : ds_copy)
            {
                mult *= el;
            }
            int dest = ds[ix];
            // Si mult==0 el producto total es 0 (válido), no hay riesgo de overflow
            if (mult == 0) continue;
            if (dest > std::numeric_limits<int>::max() / mult)
            {
                std::string error_ms =
                    "ERROR in TensorResult constructor: dims out of <int> bounds";
                std::cerr << error_ms << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        T *ptr;
        size_t siz = dims.getTotal() * sizeof(T);

        if (space == MemorySpace::Device)
        {
            CHECK_CUDA(cudaMalloc(&ptr, siz));
        }
        else
        {
            ptr = static_cast<T *>(std::malloc(siz));
        }

        data = ptr;
    }

    // Constructor por defecto
    TensorResult() : data(nullptr), space(MemorySpace::Host), dims({0, 0, 0, 0}) {}

    TensorResult(py::object obj)
    {
        // Obtener capsule desde __dlpack__()
        py::object capsule_obj = obj.attr("__dlpack__")();
        py::capsule capsule = capsule_obj.cast<py::capsule>();

        if (std::string(capsule.name()) != "dltensor")
        {
            throw std::runtime_error("Object does not provide a valid DLPack tensor");
        }

        DLManagedTensor *managed = capsule.get_pointer<DLManagedTensor>();

        if (!managed)
        {
            throw std::runtime_error("Null DLManagedTensor");
        }

        auto &dl = managed->dl_tensor;

        if (dl.ndim < 3)
        {
            throw std::runtime_error("Expected tensor with ndim >= 3");
        }

        int b = dl.shape[0];
        int m = dl.shape[1];
        int n = dl.shape[2];

        dims = {b, m, n, 1};

        auto dev = dl.device.device_type;
        space = (dev == kDLCUDA) ? MemorySpace::Device : MemorySpace::Host;

        size_t bytes = b * m * n * sizeof(T);

        T *new_data;

        if (dev == kDLCUDA)
        {
            CHECK_CUDA(cudaMalloc(&new_data, bytes));
            CHECK_CUDA(cudaMemcpy(new_data, dl.data, bytes, cudaMemcpyDeviceToDevice));
        }
        else
        {
            new_data = (T *)std::malloc(bytes);
            std::memcpy(new_data, dl.data, bytes);
        }

        

        data = new_data;

        released = false;

        // 🔥 Consumir el dlpack original
        managed->deleter(managed);

        // Marcar capsule como consumida (estándar dlpack)
        PyCapsule_SetName(capsule.ptr(), "used_dltensor");
    }

    ~TensorResult()
    {

        if (released || !data)
        {
            return;
        }

        if (managed)
        {

            // managed->deleter(managed);
            managed = nullptr;
            return;
        }

        if (space == MemorySpace::Device)
        {
            CHECK_CUDA(cudaFree(data));
        }
        else

        {
            std::free(data);
        }
    }

    // Función para obtener el tamaño en bytes
    size_t size_bytes()
    {
        return dims.getTotal() * sizeof(T);
    }

    // Getters para compatibilidad con código antiguo
    T *getData() const
    {
        return data;
    }
    int getBatch() const
    {
        return dims.b;
    }
    int getM() const
    {
        return dims.m;
    }
    int getN() const
    {
        return dims.n;
    }
    int getK() const
    {
        return dims.k;
    }

    bool isDevicePtr() const
    {
        return space == MemorySpace::Device;
    }
    MemorySpace getMemorySpace() const
    {
        return space;
    }

    TensorResult<T> clone()
    {
        TensorResult<T> out(space, dims.b, dims.m, dims.n, dims.k, true);

        if (space == MemorySpace::Device)
        {
            CHECK_CUDA(cudaMemcpy(out.data, data, size_bytes(), cudaMemcpyDeviceToDevice));
        }
        else
            std::memcpy(out.data, data, size_bytes());

        return out;
    }

    // Función para obtener el número total de elementos
    int total_elements()
    {
        return dims.getTotal();
    }

    void move_to_device()
    {
        if (space == MemorySpace::Device)
            return;

        T *dev;
        CHECK_CUDA(cudaMalloc(&dev, size_bytes()));
        CHECK_CUDA(cudaMemcpy(dev, data, size_bytes(), cudaMemcpyHostToDevice));

        std::free(data);
        data = dev;
        space = MemorySpace::Device;
    }

    void move_to_host()
    {
        if (space == MemorySpace::Host)
            return;

        T *host;
        host = (T *)malloc(size_bytes());
        CHECK_CUDA(cudaMemcpy(host, data, size_bytes(), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(data));
        data = host;
        space = MemorySpace::Host;
    }

    py::capsule __dlpack__()
    {
        if (released)
        {
            throw std::runtime_error("Tensor already exported to DLPack");
        }

        DLManagedTensor *managed = new DLManagedTensor();

        managed->dl_tensor.data = data;

        managed->dl_tensor.device = {space == MemorySpace::Device ? kDLCUDA : kDLCPU, 0};

        managed->dl_tensor.ndim = 4;

        int64_t *shape = new int64_t[4]{dims.b, dims.m, dims.n, 
          dims.k};
        managed->dl_tensor.shape = shape;

        int64_t *strides = make_c_strides({dims.b, dims.m, dims.n, dims.k});
        managed->dl_tensor.strides = strides;

        managed->dl_tensor.byte_offset = 0;

        managed->dl_tensor.dtype = {kDLBfloat, 16, 1};

        // 🔥 guardar puntero al tensor
        managed->manager_ctx = this;

        managed->deleter = [](DLManagedTensor *self)
        {
            TensorResult *tensor = static_cast<TensorResult *>(self->manager_ctx);

            delete[] self->dl_tensor.shape;
            std::free(self->dl_tensor.strides);

            delete tensor; // 🔥 destruye el TensorResult
            delete self;
        };

        released = true;

        return py::capsule(managed, "dltensor");
    }

    py::object __dlpack_device__() const
    {
        if (space == MemorySpace::Device)
        {
            return py::make_tuple(kDLCUDA, 0);
        }
        else
        {
            return py::make_tuple(kDLCPU, 0);
        }
    }

    std::vector<T> to_host_vector() const
    {
        size_t total = dims.getTotal();
        std::vector<T> host(total);

        if (space == MemorySpace::Device)
        {
            CHECK_CUDA(cudaMemcpy(host.data(), data, total * sizeof(T), cudaMemcpyDeviceToHost));
        }
        else
        {
            std::memcpy(host.data(), data, total * sizeof(T));
        }

        return host;
    }

    void debug_print()
    {
        auto v = to_host_vector();
        for (int i = 0; i < std::min(10, (int)v.size()); ++i)
            std::cout << static_cast<float>(v[i]) << " ";
        if ((int)v.size() > 10)
        {
            std::cout << " ...";
        }
        std::cout << "\n";
    }
};

inline unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#endif
