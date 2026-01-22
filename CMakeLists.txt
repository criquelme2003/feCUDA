cmake_minimum_required(VERSION 3.28)

project(FeCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.1/bin/nvcc)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

file(GLOB ALGORITHM_SOURCES "src/algorithms/*.cu")
file(GLOB UTILS_SOURCES "src/utils/*.cu")
file(GLOB_RECURSE KERNEL_SOURCES "src/kernels/*.cu")

# ===== Ejecutable =====
add_executable(fecuda_main
  src/main.cu
  ${ALGORITHM_SOURCES}
  ${UTILS_SOURCES}
  ${KERNEL_SOURCES}
)

target_include_directories(fecuda_main
  PRIVATE include
  /home/carlos/dlpack/include
)

target_link_libraries(fecuda_main
  CUDA::curand
  Python::Python

)

# ===== Módulo Python =====
pybind11_add_module(forgethreads
  src/bindings.cu
  ${ALGORITHM_SOURCES}
  ${UTILS_SOURCES}
  ${KERNEL_SOURCES}
)

target_include_directories(forgethreads
  PRIVATE include
  /home/carlos/dlpack/include

)
target_link_libraries(forgethreads
  PRIVATE
  CUDA::curand
  Python::Python
)

install(
  TARGETS forgethreads
  LIBRARY DESTINATION lib/python3.12/site-packages

)
