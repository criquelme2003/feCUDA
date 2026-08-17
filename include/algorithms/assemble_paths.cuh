#ifndef ASSEMBLE_PATHS_CUH
#define ASSEMBLE_PATHS_CUH

#include <cuda_fp16.h>
#include <utility>
#include <vector>

// Par (paths, values) devuelto por assemble_paths.
using PathsAndValues = std::pair<std::vector<std::vector<int>>, std::vector<float>>;

// Ensambla o extiende caminos usando el argmax que ya calculó el kernel en GPU.
//
// Primera llamada (prev_paths vacío):
//   Para cada (b,m,n) donde C[m,n] - A[m,n] >= thr, el pivote k viene
//   directamente de argmax[b,m,n]. Devuelve paths [b, m, k, n].
//
// Llamadas siguientes:
//   Para cada path [b,m,...,n], busca n2 donde C[m,n2] - A[m,n2] >= thr
//   y argmax[b,m,n2] == n (el kernel eligió n como pivote óptimo).
//   Devuelve paths extendidos con n2 al final.
//
// A = C_{s-1},  C = C_s,  argmax = resultado del kernel para este step.
//
// Compartida por todas las variantes de maxmin (v1/v2/v3): una única definición
// en src/algorithms/assemble_paths.cu evita colisiones de símbolo en el link.
PathsAndValues assemble_paths(
    std::vector<std::vector<int>> prev_paths,
    __half *d_A,
    __half *d_C,
    int *d_argmax,
    float thr,
    int M,
    int N,
    int B
);

#endif
