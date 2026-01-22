#include "../../include/core/types.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#define MAX_GRID_SIZE 10000
#define MAX_PATHS_PER_ITER 5000000

__global__ void checkExist2(__half *A, const __half *B, const __half *Cmax,
                            const __half *Cmin, int K, int te)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int id = t; id < te; id += stride)
  {

    int check =
        __half2int_rn(A[id]) + __half2int_rn(B[id]) + __half2int_rn(Cmax[id]);

    for (int k = 0; k < K && id + k < te; k++)
    {
      check += __half2int_rn(Cmin[id + k]);
    }
    A[id] = __half(check);
  }
}

template <typename T>
std::vector<std::tuple<int4 *, T *>> maxmin(TensorResult<T> &tensor1, TensorResult<T> &tensor2, T thr, int order)
{

  // Validar que los tensores sean 3D (K=1) como espera el kernel (WRAPPER
  // MAKES)

  std::vector<std::tuple<int4 *, T *>> ret;

  if (tensor1.getK() != 1 || tensor2.getK() != 1)
  {
    printf("Error: maxmin solo acepta tensores 3D (K=1)\n");
    exit(0); // tensor nulo
  }

  // Para el kernel, necesitamos que A sea [batch, M, K] y B sea [batch, K, N]
  // Pero como K=1, efectivamente son [batch, M] y [batch, N]
  int B = tensor1.getBatch();
  int M = tensor1.getM();
  int K = tensor1.getN(); // En el contexto del kernel, N del tensor1 es K
  int N = tensor2.getN();

  // Alocar memoria en device
  T *d_A, *d_B, *d_C_min, *d_C_max;

  tensor1.move_to_device();
  tensor2.move_to_device();
  // max_result.move_to_device();
  // min_result.move_to_device();

  // d_C_min = max_result.getData();
  // d_C_max = min_result.getData();

  // CHECK_CUDA(cudaEventRecord(start));
  int *d_counter;

  dim3 block(128); // >= K
  dim3 grid(N, M, B);
  size_t shmem = 128 * sizeof(T);
  cudaMalloc(&d_counter, sizeof(int));
  cudaMemset(d_counter, 0, sizeof(int));

  int4 *paths;
  T *values;

  cudaMalloc(&paths, MAX_PATHS_PER_ITER * sizeof(int4));
  cudaMalloc(&values, MAX_PATHS_PER_ITER * sizeof(T));

  if ((B * M * N * K) < MAX_PATHS_PER_ITER)
  {
    maxmin_threshold_kernel<T>
        <<<grid, block, shmem>>>(
            tensor1.getData(),
            tensor2.getData(),
            paths,
            values,
            d_counter,
            thr,
            B, M, N, K);
  }
  else
  {
    d_A = tensor1.getData();
    d_B = tensor2.getData();

    int sizeA = M * K;
    int sizeB = K * N;
    for (int b_ = 0; b_ <= B; b_++)
    {
      T *localA = d_A * (b_ * sizeA);
      T *localB = d_B * (b_ * sizeB);
      


    }
  }

  CHECK_CUDA(cudaDeviceSynchronize())
  // CHECK_CUDA(cudaEventRecord(end));
  ret.push_back(std::make_tuple(paths, values));
  return ret;
}

template std::vector<std::tuple<int4 *, float *>> maxmin(TensorResult<float> &tensor1, TensorResult<float> &tensor2, float thr, int order);

template std::vector<std::tuple<int4 *, __half *>> maxmin(TensorResult<__half> &tensor1, TensorResult<__half> &tensor2, __half thr, int order);