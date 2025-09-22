#include <utils.cuh>
#include <utils/file_io.cuh>
#include <core/types.cuh>
#include <headers.cuh>
#include <vector>
#include <fstream>

void iterative_test(

)
{

    TensorResult t1;
    leer_matriz_3d_desde_archivo("../datasets_txt/CC.txt", t1, 10, 16, 16);

    std::vector<TensorResult> paths;
    std::vector<TensorResult> values;
    std::vector<TensorResult> pure_paths;
    std::vector<TensorResult> pure_values;

    iterative_maxmin_cuadrado(t1, 0.3, 3, paths, values, pure_paths, pure_values);
    
    imprimir_tensor(paths[1]);
    save_tensor_vector(paths, "vecpaths.txt");
    save_tensor_vector(values, "vecvalues.txt");
}