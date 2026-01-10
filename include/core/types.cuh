#ifndef TYPES_CUH
#define TYPES_CUH

#include "../../include/utils.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits.h>
#include <limits>
#include <vector>

enum class MemorySpace { Host, Device };

struct TensorResultDims {
  int b = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  int getTotal() { return b * m * n * k; }
};

// Estructura para mantener información completa del tensor
template <typename T = float> struct TensorResult {
private:
  T *data; // Puntero a los datos
  TensorResultDims
      dims; // Dimensiones del tensor (K para dimensiones adicionales)
  MemorySpace space;

public:
  // Constructor completo con ownership
  TensorResult(MemorySpace memory_space, int b, int m, int n, int k = 1,
               bool owns = true)
      : data(nullptr), space(memory_space), dims({b, m, n, k}) {
    allocateData();
  }

  // Reservar memoria para el tensor
  void allocateData() {

    // Controlar dimensiones
    std::vector<int> ds = {dims.b, dims.m, dims.n, dims.k};

    for (int ix = 0; ix < ds.size(); ix++) {
      std::vector<int> ds_copy(ds);

      ds_copy.erase(ds_copy.begin() + ix);

      int mult = 1;
      for (auto el : ds_copy) {
        mult *= el;
      }
      int dest = ds[ix];
      if (dest > std::numeric_limits<int>::max() / mult) {
        std::string error_ms =
            "ERROR in TensorResult constructor: dims out of <int> bounds";
        std::cerr << error_ms << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    T *ptr;
    size_t siz = dims.getTotal() * sizeof(T);

    if (space == MemorySpace::Device) {
      CHECK_CUDA(cudaMalloc(&ptr, siz));
    } else {
      ptr = static_cast<T *>(std::malloc(siz));
    }

    data = ptr;
  }

  // Constructor por defecto
  TensorResult()
      : data(nullptr), space(MemorySpace::Host), dims({0, 0, 0, 0}) {}

  ~TensorResult() {
    if (!data)
      return;

    if (space == MemorySpace::Device) {
      CHECK_CUDA(cudaFree(data));
    }

    else {
      std::free(data);
    }
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

  TensorResult clone() const {
    TensorResult out(space, dims.b, dims.m, dims.n, dims.k, true);

    if (space == MemorySpace::Device) {
      CHECK_CUDA(
          cudaMemcpy(out.data, data, size_bytes(), cudaMemcpyDeviceToDevice));
    } else
      std::memcpy(out.data, data, size_bytes());

    return out;
  }
  TensorResult(const TensorResult &) = delete;
  TensorResult &operator=(const TensorResult &) = delete;

  // Función para obtener el número total de elementos
  int total_elements() { return dims.getTotal(); }

  void move_to_device() {
    if (space == MemorySpace::Device)
      return;

    T *dev;
    CHECK_CUDA(cudaMalloc(&dev, size_bytes()));
    CHECK_CUDA(cudaMemcpy(dev, data, size_bytes(), cudaMemcpyHostToDevice));

    std::free(data);
    data = dev;
    space = MemorySpace::Device;
  }

  void move_to_host() {
    if (space == MemorySpace::Host)
      return;

    T *host;
    host = static_cast<T *>(size_bytes());
    CHECK_CUDA(cudaMemcpy(host, data, size_bytes(), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(data));
    data = host;
    space = MemorySpace::Host;
  }
};

inline unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#endif
