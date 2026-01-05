cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
endif()

project(FeCUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(CUDAToolkit REQUIRED)

file(GLOB MAIN_SOURCES "src/main.cu")
file(GLOB ALGORITHM_SOURCES "src/algorithms/*.cu")
file(GLOB UTILS_SOURCES "src/utils/*.cu")
file(GLOB_RECURSE KERNEL_SOURCES "src/kernels/*.cu")

add_executable(fecuda_main
    ${MAIN_SOURCES}
    ${ALGORITHM_SOURCES}
    ${UTILS_SOURCES}
    ${KERNEL_SOURCES}
)

target_link_libraries(fecuda_main
    CUDA::curand
)

target_compile_definitions(fecuda_main PUBLIC FOO=1)


set_target_properties(fecuda_main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

target_compile_options(fecuda_main PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-G>
)

target_compile_options(fecuda_main PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -lineinfo>
)