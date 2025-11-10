#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

namespace
{
float time_ms(const std::function<void()> &fn, int reps, int warmup)
{
    for (int i = 0; i < warmup; ++i)
        fn();

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float acc = 0.0f;
    for (int i = 0; i < reps; ++i)
    {
        cudaEventRecord(start);
        fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        acc += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return acc / static_cast<float>(reps);
}

__global__ void noop_kernel()
{
    // kernel vacío para ejemplo
}
} // namespace

int main(int argc, char **argv)
{
    int reps = 40;
    int warmup = 5;

    if (argc > 1)
        reps = std::atoi(argv[1]);
    if (argc > 2)
        warmup = std::atoi(argv[2]);

    float avg_ms = time_ms([]() { noop_kernel<<<1, 1>>>(); }, reps, warmup);
    printf("noop_kernel avg_ms=%.6f reps=%d warmup=%d\n", avg_ms, reps, warmup);
    return 0;
}
