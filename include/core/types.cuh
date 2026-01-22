#ifndef TYPES_CUH
#define TYPES_CUH
#include "../../include/utils.cuh"
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
#include "dlpack/dlpack.h"

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
  int getTotal() { return b * m * n * k; }
};

inline int64_t *make_c_strides(const std::vector<int64_t> &shape)
{
  int ndim = shape.size();
  int64_t *strides = static_cast<int64_t *>(
      std::malloc(ndim * sizeof(int64_t)));

  int64_t stride = 1;
  for (int i = ndim - 1; i >= 0; --i)
  {
    strides[i] = stride;
    stride *= shape[i];
  }

  return strides;
}

// Estructura para mantener información completa del tensor
template <typename T = float>
struct TensorResult
{
private:
  T *data; // Puntero a los datos
  TensorResultDims
      dims; // Dimensiones del tensor (K para dimensiones adicionales)
  MemorySpace space;
  bool released = false;

public:
  // Constructor completo con ownership
  TensorResult(MemorySpace memory_space, int b, int m, int n, int k = 1,
               bool owns = true)
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
  TensorResult()
      : data(nullptr), space(MemorySpace::Host), dims({0, 0, 0, 0}) {}
  ~TensorResult()
  {
    if (!data || released)
      return;

    if (space == MemorySpace::Device)
    {
      CHECK_CUDA(cudaFree(data));
    }
    else
    {
      std::free(data);
    }
    std::cout << "Deleted from tensor class" << std::endl;
    }

  // Función para obtener el tamaño en bytes
  size_t size_bytes() { return dims.getTotal() * sizeof(T); }

  // Getters para compatibilidad con código antiguo
  T *getData() const { return data; }
  int getBatch() const { return dims.b; }
  int getM() const { return dims.m; }
  int getN() const { return dims.n; }
  int getK() const { return dims.k; }

  bool isDevicePtr() const { return space == MemorySpace::Device; }
  MemorySpace getMemorySpace() const { return space; }

  TensorResult<T> clone()
  {
    TensorResult<T> out(space, dims.b, dims.m, dims.n, dims.k, true);

    if (space == MemorySpace::Device)
    {
      CHECK_CUDA(
          cudaMemcpy(out.data, data, size_bytes(), cudaMemcpyDeviceToDevice));
    }
    else
      std::memcpy(out.data, data, size_bytes());

    return out;
  }

  // Función para obtener el número total de elementos
  int total_elements() { return dims.getTotal(); }

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
      throw std::runtime_error("Tensor already consumed by DLPack");
    }

    // 1️⃣ Alloc DLManagedTensor dinámicamente
    DLManagedTensor *managed = new DLManagedTensor();

    // 2️⃣ Contexto
    managed->dl_tensor.data = data;
    std::cout << "DATA[0] from __dlpack__()" <<  data[0] << std::endl;
    
    managed->dl_tensor.device = {
        space == MemorySpace::Device ? kDLCUDA : kDLCPU,
        0};

    managed->dl_tensor.ndim = 4;

    // shape dinámico
    int64_t *shape = new int64_t[4]{
        dims.b, dims.m, dims.n, dims.k};
    managed->dl_tensor.shape = shape;

    managed->dl_tensor.strides =
        (int64_t *)std::malloc(4 * sizeof(int64_t));

    managed->dl_tensor.strides[0] = dims.m * dims.n * dims.k;
    managed->dl_tensor.strides[1] = dims.n * dims.k;
    managed->dl_tensor.strides[2] = dims.k;
    managed->dl_tensor.strides[3] = 1;
    // contiguous
    managed->dl_tensor.byte_offset = 0;

    // dtype
    managed->dl_tensor.dtype = {
        kDLFloat,
        32,// static_cast<uint8_t>(sizeof(T) * 8),
        1};

    // 3️⃣ Deleter (SE EJECUTA CUANDO NUMPY TERMINA)
    managed->deleter = [](DLManagedTensor *self)
    {
      if (!self)
        return;

      if (self->dl_tensor.device.device_type == kDLCUDA)
      {
        cudaFree(self->dl_tensor.data);
      }
      else
      {
        std::free(self->dl_tensor.data);
      }

      delete[] self->dl_tensor.shape;
      delete self;
      std::cout << "Deleted from managed" << std::endl;
    };

    // 4️⃣ Transferimos ownership
    data = nullptr;
    released = true;

    // 5️⃣ Capsule con NOMBRE CORRECTO
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
