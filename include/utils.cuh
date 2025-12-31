#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <core/types.cuh>
#include <cstdlib>
#include <iostream>
#include <string>
#include <curand.h>
#include <vector>

// Macro para verificar errores de CUDA con exceptions
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

#define CHECK_CURAND(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


// Función para limpiar memoria de TensorResult de forma segura
void safe_tensor_cleanup(TensorResult &tensor);

// Función para crear una copia del tensor en memoria host
TensorResult copy_tensor(const TensorResult &src);
TensorResult copy_tensor_to_cpu(const TensorResult &src);
TensorResult copy_tensor_to_gpu(const TensorResult &src);

void calculate_prima(const TensorResult &maxmin_conjugado, const TensorResult &gen_tensor,TensorResult &prima, bool keep_in_device = false);

// template<typename F1,typename F2,typename... Op_args>
// void batchedOp( 
//     CutEdge cut_edge,
//     F1 op, 
//     F2 get_gpu_size_req,
//     F2 get_cpu_size_req,
//     Op_args... args
// ){
//     size_t *free_gpu_mem, *_;
//     size_t req_gpu_mem;
    
//     CHECK_CUDA(cudaMemGetInfo(free_gpu_mem, _));

//     req_gpu_mem = get_gpu_size_req(args...);

//     std::cout << "free_gpu_mem: " << free_gpu_mem << std::endl;
//     std::cout << "req_gpu_mem: " << req_gpu_mem << std::endl;

// };


enum class CutEdge
{
    BATCH,   // Significa que la operacion es independiente cada MXNXK
    ELEMENT, // Significa que la operacion es interdependiente en el eje BATCH
};


template<typename... Op_args>
void batchedOp( 
    std::vector<CutEdge> cut_edges,
    void (*op)(Op_args...), 
    size_t (*get_gpu_size_req)(Op_args...),
    size_t (*get_cpu_size_req)(Op_args...),
    Op_args... args
){
    size_t free_gpu_mem, _;
    size_t req_gpu_mem;
    
    CHECK_CUDA(cudaMemGetInfo(&free_gpu_mem, &_));

    req_gpu_mem = get_gpu_size_req(args...);

    std::cout << "free_gpu_mem: " << free_gpu_mem << std::endl;
    std::cout << "req_gpu_mem: " << req_gpu_mem << std::endl;
    op(args...);

    if(req_gpu_mem < free_gpu_mem){
        // specific cuts by arg (only TensorResult types)
        int count = 0;
        for (const auto p : {args...})
        {
            
            
            if(count == cut_edges.size())
                break;
        }
    }


};

#endif
