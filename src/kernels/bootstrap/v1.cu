#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <ctime>
#include <core/types.cuh>

void FEempirical(const TensorResult &tensor1, TensorResult &result2, int rep)
{

    // Get the size of the input tensor
    int size = tensor1.M * tensor1.N * tensor1.batch;

    // Create a device vector to hold the input data
    thrust::device_vector<float> d_input(tensor1.data, tensor1.data + size);

    // Create a device vector to hold the output data
    thrust::device_vector<float> d_output(size);

    // Create a random number generator
    thrust::default_random_engine rng;

    // Perform the empirical bootstrap resampling
    for (int i = 0; i < rep; i++)
    {
        // Shuffle the input data
        thrust::shuffle(d_input.begin(), d_input.end(), rng);

        // Compute the mean of the shuffled data
        float mean = thrust::reduce(d_input.begin(), d_input.end(), 0.0f) / size;

        // Store the result
        d_output[i] = mean;
    }

    // Copy the output data to the result tensor

    thrust::copy(d_output.begin(), d_output.end(), result2.data);
}