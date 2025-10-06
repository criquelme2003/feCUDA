#ifndef VALIDATION_UTILS_CUH
#define VALIDATION_UTILS_CUH

#include <iostream>
#include <core/types.cuh>

namespace ValidationUtils
{

    // Validador simple para entradas
    struct InputValidator
    {
        static bool validate_paths_input(const TensorResult &previous_paths,
                                         const TensorResult &result_tensor,
                                         const TensorResult &result_values)
        {
            if (previous_paths.data == nullptr)
            {
                std::cerr << "Error: previous_paths es nulo\n";
                return false;
            }
            if (result_tensor.data == nullptr)
            {
                std::cerr << "Error: result_tensor es nulo\n";
                return false;
            }
            if (result_values.data == nullptr)
            {
                std::cerr << "Error: result_values es nulo\n";
                return false;
            }
            return true;
        }

        static bool validate_dimensions(int num_current_tensor, int num_values)
        {
            if (num_current_tensor != num_values)
            {
                std::cerr << "Error: Número de elementos en result_tensor (" << num_current_tensor
                          << ") no coincide con result_values (" << num_values << ")\n";
                return false;
            }
            return true;
        }

        // Validación general para tensores no nulos
        static bool validate_tensor_not_null(const TensorResult &tensor, const char *tensor_name = "Tensor")
        {
            if (tensor.data == nullptr)
            {
                std::cerr << "Error: " << tensor_name << " es nulo\n";
                return false;
            }
            return true;
        }

        // Validación de dimensiones compatibles entre tensores
        static bool validate_tensor_dimensions(const TensorResult &tensor1, const TensorResult &tensor2,
                                               const char *tensor1_name = "Tensor1",
                                               const char *tensor2_name = "Tensor2",
                                               bool check_exact_match = true)
        {
            if (check_exact_match)
            {
                if (tensor1.M != tensor2.M || tensor1.N != tensor2.N || tensor1.K != tensor2.K)
                {
                    std::cerr << "Error: Dimensiones incompatibles entre " << tensor1_name
                              << " (" << tensor1.M << "x" << tensor1.N << "x" << tensor1.K << ") y "
                              << tensor2_name << " (" << tensor2.M << "x" << tensor2.N << "x" << tensor2.K << ")\n";
                    return false;
                }
            }
            return true;
        }

        // Validación de rango de valores
        static bool validate_positive_dimensions(int M, int N, int K = 1)
        {
            if (M <= 0 || N <= 0 || K <= 0)
            {
                std::cerr << "Error: Las dimensiones deben ser positivas. Recibido: M=" << M
                          << ", N=" << N << ", K=" << K << "\n";
                return false;
            }
            return true;
        }
    };

} // namespace ValidationUtils

#endif // VALIDATION_UTILS_CUH
