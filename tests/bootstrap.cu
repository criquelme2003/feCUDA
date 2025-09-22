#include <bootstrap.cuh>
#include <iostream>


void test_bootstrap()

{
    int M = 2, N = 2, batch_size = 20, replicas = 10;

    // Crear tensor de datos de prueba
    float *data = (float *)malloc(M * N * batch_size * sizeof(float));
    float *out_data = (float *)malloc(M * N * replicas * sizeof(float));

    for (int i = 0; i < M * N * batch_size; i++)
    {
        data[i] = (float) rand() / RAND_MAX;
    }

    clock_t start_time = clock();
    float *bootstrap_data = bootstrap_wrapper(data, M, N, batch_size, replicas);
    clock_t end_time = clock();

    cudaMemcpy(out_data, bootstrap_data, M * N * replicas * sizeof(float), cudaMemcpyDeviceToHost);

    // Imprimir original

    printf("Original\n");
    for (int b = 0; b < batch_size; b++)
    {
        printf("batch %d\n", b);
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                printf("%.4f ", data[b * M * N + m * N + n]);
            }
            printf("\n");
        }
    }

    printf("Replicas\n");
    for (int b = 0; b < replicas; b++)
    {
        printf("batch %d\n", b);
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                printf("%.4f ", out_data[b * M * N + m * N + n]);
            }
            printf("\n");
        }
    }

    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Tiempo total de ejecución: " << elapsed_time * 1000 << " ms\n";
}