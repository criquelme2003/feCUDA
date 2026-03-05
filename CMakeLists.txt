cmake_minimum_required(VERSION 3.22)

set(EXPORT_COMPILE_COMMANDS ON)
project(FeCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.5/bin/nvcc)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(Python_EXECUTABLE /usr/bin/python3)

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
  /workspace/extern/dlpack/include
)

target_link_libraries(fecuda_main
  CUDA::curand
  Python::Python

)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

target_compile_options(fecuda_main
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
  -G
  >
)

target_compile_options(fecuda_main
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
  -g
  -O0
  >
)


# ===== Módulo Python =====
pybind11_add_module(forgethreads
  src/bindings.cu
  ${ALGORITHM_SOURCES}
  ${UTILS_SOURCES}
  ${KERNEL_SOURCES}
)

set_target_properties(forgethreads PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)


target_include_directories(forgethreads
  PRIVATE include
  /workspace/extern/dlpack/include

)
target_link_libraries(forgethreads
  PRIVATE
  CUDA::curand
  Python::Python
)

#install(
#  TARGETS forgethreads
#  LIBRARY DESTINATION /lib/python3/site-packages
#)

install(
  TARGETS fecuda_main
  LIBRARY DESTINATION /lib/python3/site-packages
)
