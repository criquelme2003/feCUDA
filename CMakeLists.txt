cmake_minimum_required(VERSION 3.18)

set(FECUDA_FALLBACK_INCLUDE "${CMAKE_SOURCE_DIR}/toolchain/fallback")
set(CMAKE_CUDA_FLAGS_INIT "-Xcompiler=-idirafter -Xcompiler=${FECUDA_FALLBACK_INCLUDE}")

if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda-13.0/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc" CACHE FILEPATH "" FORCE)
    set(CUDACXX "/usr/local/cuda-13.0/bin/nvcc" CACHE FILEPATH "" FORCE)
endif()
if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda-13.0")
    set(CUDAToolkit_ROOT "/usr/local/cuda-13.0" CACHE PATH "" FORCE)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-13.0" CACHE PATH "" FORCE)
    set(ENV{CUDA_PATH} "/usr/local/cuda-13.0")
endif()

project(FeCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

find_package(CUDAToolkit REQUIRED)

set(FECUDA_CORE_SOURCES
    src/core/tensor.cu
    src/algorithms/armar_caminos.cu
    src/algorithms/indices.cu
    src/algorithms/iterative_maxmin_cuadrado.cu
    src/algorithms/maxmin.cu
    src/algorithms/prima.cu
    src/kernels/maxmin/kernel_v1.cu
    src/utils/compare.cu
    src/utils/cuda_utils.cu
    src/utils/file_io.cu
    src/utils/imprimir_tensor.cu
    src/utils/leer_archivo.cu
    src/utils/validar_algoritmos_maxmin.cu
)

add_library(fecuda_core STATIC ${FECUDA_CORE_SOURCES})
target_include_directories(fecuda_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(fecuda_core PRIVATE /usr/include)
target_link_libraries(fecuda_core PUBLIC CUDA::cudart)
set_target_properties(fecuda_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(fecuda_main src/main.cu)
target_link_libraries(fecuda_main PRIVATE fecuda_core)
target_compile_definitions(fecuda_main PRIVATE FECUDA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(fecuda_main PRIVATE /usr/include)

add_executable(fecuda_validator validation/generate_results.cu)
target_link_libraries(fecuda_validator PRIVATE fecuda_core)
target_compile_definitions(fecuda_validator PRIVATE FECUDA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(fecuda_validator PRIVATE /usr/include)
