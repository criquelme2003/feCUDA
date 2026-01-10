#include "../../include/core/types.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define MAX_GRID_SIZE 10000

/*

TODO: LIMITAR GRID SIZE Y ADAPTAR WRAPPER PARA BATCH_PROCESSING.
*/

// Versión mejorada de maxmin que usa TensorResult<> y retorna tanto max como
// min

long long ceil_div_128(__int128 n, long long d) {
  return (long long)((n + d - 1) / d);
}

template <typename T>
void maxmin(TensorResult<T> &tensor1, TensorResult<T> &tensor2,
            TensorResult<T> &max_result, TensorResult<T> &min_result,
            cudaEvent_t &start, cudaEvent_t &end, bool keep_in_device) {

  // Validar que los tensores sean 3D (K=1) como espera el kernel (WRAPPER
  // MAKES)

  if (tensor1.getK() != 1 || tensor2.getK() != 1) {
    printf("Error: maxmin solo acepta tensores 3D (K=1)\n");
    exit(0); // tensor nulo
  }

  // Para el kernel, necesitamos que A sea [batch, M, K] y B sea [batch, K, N]
  // Pero como K=1, efectivamente son [batch, M] y [batch, N]
  int batch = tensor1.getBatch();
  int M = tensor1.getM();
  int K = tensor1.getN(); // En el contexto del kernel, N del tensor1 es K
  int N = tensor2.getN();

  // Alocar memoria en device
  T *d_A, *d_B, *d_C_min, *d_C_max;

  tensor1.move_to_device();
  tensor2.move_to_device();
  max_result.move_to_device();
  min_result.move_to_device();

  d_A = tensor1.getData();
  d_B = tensor2.getData();
  d_C_min = max_result.getData();
  d_C_max = min_result.getData();

  // Copiar datos al device

  constexpr int WARPS_PER_BLOCK = 4;
  int blockSize = WARPS_PER_BLOCK * 32;
  int k_launch_size = (static_cast<int>((K + 31) / 32)) * 32;

  __int128 total =
      (__int128)M * (__int128)N * (__int128)batch * (__int128)k_launch_size;

  int gridSize = ceil_div_128(total, blockSize);

  // Ejecutar kernel
  CHECK_CUDA(cudaEventRecord(start));
  cub_max_min_kernel<T, WARPS_PER_BLOCK>
      <<<gridSize, blockSize>>>(d_A, d_B, d_C_min, d_C_max, M, K, N, batch);

  CHECK_CUDA(cudaDeviceSynchronize())
  CHECK_CUDA(cudaEventRecord(end));

  CHECK_KERNEL()

  return;
}

template void maxmin<float>(TensorResult<float> &, TensorResult<float> &,
                            TensorResult<float> &, TensorResult<float> &,
                            cudaEvent_t &, cudaEvent_t &, bool);

template void maxmin<__half>(TensorResult<__half> &, TensorResult<__half> &,
                             TensorResult<__half> &, TensorResult<__half> &,
                             cudaEvent_t &, cudaEvent_t &, bool);