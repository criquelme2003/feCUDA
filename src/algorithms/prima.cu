#include <core/types.cuh>
#include <cstdio>
#include <utils.cuh>

// Kernel para calcular prima = maxmin_conjugado - gen_tensor
__global__ void calculate_prima_kernel(float *maxmin_conjugado, float *gen_tensor,
                                       float *prima, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        prima[idx] = maxmin_conjugado[idx] - gen_tensor[idx];
    }
}

void calculate_prima(const TensorResult &maxmin_conjugado, const TensorResult &gen_tensor,
                     TensorResult &prima, bool keep_in_device)
{
    // Verificar que las dimensiones coincidan
    if (maxmin_conjugado.batch != gen_tensor.batch ||
        maxmin_conjugado.M != gen_tensor.M ||
        maxmin_conjugado.N != gen_tensor.N)
    {
        printf("Error: Dimensiones no coinciden para calcular prima\n");
        return;
    }
    
    int total_elements = maxmin_conjugado.batch * maxmin_conjugado.M * maxmin_conjugado.N;
    size_t size = total_elements * sizeof(float);
    
    float *d_maxmin_conjugado, *d_gen_tensor, *d_prima;
    
    // Alocar memoria device para prima
    CHECK_CUDA(cudaMalloc(&d_prima, size));
    
    // Copiar datos a device
    if (maxmin_conjugado.is_device_ptr)
    {
        d_maxmin_conjugado = maxmin_conjugado.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_maxmin_conjugado, size));
        CHECK_CUDA(cudaMemcpy(d_maxmin_conjugado, maxmin_conjugado.data, size, cudaMemcpyHostToDevice));
    }
    
    if (gen_tensor.is_device_ptr)
    {
        d_gen_tensor = gen_tensor.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_gen_tensor, size));
        CHECK_CUDA(cudaMemcpy(d_gen_tensor, gen_tensor.data, size, cudaMemcpyHostToDevice));
    }
    
    // Lanzar kernel
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    calculate_prima_kernel<<<grid_size, block_size>>>(d_maxmin_conjugado, d_gen_tensor,
                                                      d_prima, total_elements);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Configurar TensorResult de salida según keep_in_device
    if (keep_in_device)
    {
        // Mantener en GPU
        prima.data = d_prima;
        prima.is_device_ptr = true;
        prima.owns_memory = true;
    }
    else
    {
        // Transferir a CPU
        float *h_prima = (float *)malloc(size);
        CHECK_CUDA(cudaMemcpy(h_prima, d_prima, size, cudaMemcpyDeviceToHost));
        
        prima.data = h_prima;
        prima.is_device_ptr = false;
        prima.owns_memory = true;
        
        // Liberar memoria GPU del resultado ya que se copió a CPU
        cudaFree(d_prima);
    }
    
    // Configurar dimensiones
    prima.batch = maxmin_conjugado.batch;
    prima.M = maxmin_conjugado.M;
    prima.N = maxmin_conjugado.N;
    prima.K = maxmin_conjugado.K;
    
    // Limpiar memoria device temporal
    if (!maxmin_conjugado.is_device_ptr)
        cudaFree(d_maxmin_conjugado);
    if (!gen_tensor.is_device_ptr)
        cudaFree(d_gen_tensor);
}
