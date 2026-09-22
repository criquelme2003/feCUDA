cmake_minimum_required(VERSION 4.0)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.2/bin/nvcc)

project(FeCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


find_package(CUDAToolkit REQUIRED)

#set(Python_EXECUTABLE /home/criquelme/.conda/envs/fecuda/bin/python)
#set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)


include(FetchContent)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)

#==== pybind11 =====
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6  # fija una versión estable
)

FetchContent_MakeAvailable(pybind11)
  
# ==== DLPACK ======

FetchContent_Declare(
  dlpack
  GIT_REPOSITORY https://github.com/dmlc/dlpack.git
  GIT_TAG v0.8              # o el commit/tag que necesites
  EXCLUDE_FROM_ALL
  )


FetchContent_MakeAvailable(dlpack)

# ===== cuCollections ======    
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS 
      "BUILD_TESTS OFF" 
      "BUILD_BENCHMARKS OFF" 
      "BUILD_EXAMPLES OFF" 
  
)

if(TARGET cuco)
  message(STATUS "cuco target: OK")
else()
  message(FATAL_ERROR "cuco target NO existe")
endif()

# ==== GLOBS ========
file(GLOB ALGORITHM_SOURCES "src/algorithms/*.cu")
file(GLOB UTILS_SOURCES "src/utils/*.cu")
file(GLOB_RECURSE KERNEL_SOURCES "src/kernels/*.cu")

add_executable(fecuda_main
  src/main.cu
  ${ALGORITHM_SOURCES}
  ${UTILS_SOURCES}
  ${KERNEL_SOURCES}
)

target_include_directories(fecuda_main
  PRIVATE include
  /home/criquelme/feCUDA/include
)

target_link_libraries(fecuda_main
  PRIVATE
  CUDA::curand
  pybind11::headers
  Python::Python
  cuco
  dlpack::dlpack
)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

target_compile_options(fecuda_main
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
  -lineinfo
  --expt-extended-lambda
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

)
target_link_libraries(forgethreads
  PRIVATE
  CUDA::curand
)

set_source_files_properties(src/bindings.cu PROPERTIES
  COMPILE_OPTIONS "-Xcompiler=-O1"
)




#install(
#  TARGETS forgethreads
#  LIBRARY DESTINATION /lib/python3/site-packages
#)

target_compile_options(forgethreads
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
  --expt-extended-lambda
  >
)
install(
  TARGETS fecuda_main
  LIBRARY DESTINATION /lib/python3/site-packages
)
