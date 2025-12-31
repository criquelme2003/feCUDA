#include <utils.cuh>
#include <cstdio>




// template<typename... Op_args>
// void batchedOp( 
//     CutEdge cut_edge,
//     void (*op)(Op_args...), 
//     size_t (*get_gpu_size_req)(Op_args...),
//     size_t (*get_cpu_size_req)(Op_args...),
//     Op_args... args
// ){
//     size_t *free_gpu_mem, *_;
//     size_t req_gpu_mem;
    
//     CHECK_CUDA(cudaMemGetInfo(free_gpu_mem, _));

//     req_gpu_mem = get_gpu_size_req(args...);

//     std::cout << "free_gpu_mem: " << free_gpu_mem << std::endl;
//     std::cout << "req_gpu_mem: " << req_gpu_mem << std::endl;

// };


