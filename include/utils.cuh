#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>

// Macro para verificar errores de CUDA con exceptions
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::string error_msg = std::string("CUDA error at ") + __FILE__ + ":" + \
                              std::to_string(__LINE__) + ": " +                \
                              cudaGetErrorString(err);                         \
      std::cerr << error_msg << std::endl;                                     \
      cudaDeviceReset();                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_KERNEL()                                                         \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA kernel launch error at " << __FILE__ << ":"           \
                << __LINE__ << " : " << cudaGetErrorString(err) << std::endl;  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_CURAND(x)                                                        \
  do {                                                                         \
    if ((x) != CURAND_STATUS_SUCCESS) {                                        \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

// // Función para limpiar memoria de TensorResult<> de forma segura

// enum class CutEdge
// {
//     BATCH,   // Significa que la operacion es independiente cada MXNXK
//     ELEMENT, // Significa que la operacion es interdependiente en el eje
//     BATCH
// };

// template<typename... Op_args>
// void batchedOp(
//     std::vector<CutEdge> cut_edges,
//     void (*op)(Op_args...),
//     size_t (*get_gpu_size_req)(Op_args...),
//     size_t (*get_cpu_size_req)(Op_args...),
//     Op_args... args
// ){
//     size_t free_gpu_mem, _;
//     size_t req_gpu_mem;

//     CHECK_CUDA(cudaMemGetInfo(&free_gpu_mem, &_));

//     req_gpu_mem = get_gpu_size_req(args...);

//     std::cout << "free_gpu_mem: " << free_gpu_mem << std::endl;
//     std::cout << "req_gpu_mem: " << req_gpu_mem << std::endl;
//     op(args...);

//     if(req_gpu_mem < free_gpu_mem){
//         // specific cuts by arg (only TensorResult<> types)
//         int count = 0;
//         for (const auto p : {args...})
//         {

//             if(count == cut_edges.size())
//                 break;
//         }
//     }

// };
//----------------------------------------------------------------
// template <typename T>
// TensorResult<T> batched_tensor(TensorResult<T> t, int batch_size, int
// iteration);

// extern template TensorResult<float> batched_tensor(TensorResult<float> t, int
// batch_size, int iteration);

// extern template TensorResult<__half> batched_tensor(TensorResult<__half> t,
// int batch_size, int iteration);

//----------------------------------------------------------------

#endif
