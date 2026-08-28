cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

project(FeCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(DLPACK_ROUTE "/home/criquelme/feCUDA/extern/dlpack")

find_package(CUDAToolkit REQUIRED)

set(Python_EXECUTABLE /home/criquelme/.conda/envs/fecuda/bin/python)
set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6  # fija una versión estable
)
FetchContent_MakeAvailable(pybind11)

include(FetchContent)
# ===== cuCollections ======
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  cuco
  GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
  GIT_TAG dev
)
FetchContent_MakeAvailable(cuco)

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
  ${DLPACK_ROUTE}/include
  /home/criquelme/feCUDA/include
)

target_link_libraries(fecuda_main
  PRIVATE
  CUDA::curand
  pybind11::headers
  Python::Python
  cuco
)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

target_compile_options(fecuda_main
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
  -lineinfo
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
  ${DLPACK_ROUTE}/include

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
