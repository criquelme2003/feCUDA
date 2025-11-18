cmake_minimum_required(VERSION 3.18)

# Configurar compiladores explícitamente
set(CMAKE_C_COMPILER gcc-12)
set(CMAKE_CXX_COMPILER g++-12)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)

# Forzar que NVCC use GCC-12
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin g++-12")
set(CMAKE_CUDA_HOST_COMPILER g++-12)

# Variables de entorno para asegurar el uso correcto
set(ENV{CC} gcc-12)
set(ENV{CXX} g++-12)

set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
project(FeCUDA CUDA CXX)

include_directories(/usr/include)   
# Usar C++17 para mejor soporte de características modernas
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Habilitar compilación separable CUDA globalmente
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -G -O2")

# Incluir directorios de headers - nueva estructura
include_directories(include)
include_directories(include/core)
include_directories(include/algorithms)
include_directories(include/utils)
include_directories(include/kernels)

# Directorios donde buscar bibliotecas compartidas
link_directories(/usr/lib/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/libcutensor/12)

# Archivos fuente del proyecto - nueva estructura modular
file(GLOB MAIN_SOURCES "src/main.cu")
file(GLOB CORE_SOURCES "src/core/*.cu")
file(GLOB ALGORITHM_SOURCES "src/algorithms/*.cu")
file(GLOB UTILS_SOURCES "src/utils/*.cu")
file(GLOB_RECURSE KERNEL_SOURCES "src/kernels/*.cu")
file(GLOB KERNEL_MAXMIN_SOURCES "src/kernels/maxmin/*.cu")
file(GLOB KERNEL_UTILS_SOURCES "src/kernels/utils/*.cu")



# Archivos para targets separados
file(GLOB BENCHMARK_SOURCES "benchmarks/*.cu")
file(GLOB EXAMPLE_SOURCES "examples/*.cu")
file(GLOB VALIDATION_SOURCES "validation/*.cu")
set(SERVICE_BINDING_SOURCE "services/effects_api_binding.cpp")

# Mostrar qué archivos se están compilando (para debug)
message(STATUS "Archivos principales: ${MAIN_SOURCES}")
message(STATUS "Archivos core: ${CORE_SOURCES}")
message(STATUS "Archivos algorithms: ${ALGORITHM_SOURCES}")
message(STATUS "Archivos utils: ${UTILS_SOURCES}")
message(STATUS "Archivos kernels: ${KERNEL_SOURCES}")

set(mathdx_ROOT "$ENV{HOME}/nvidia/mathdx/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
enable_language(CUDA)
find_package(CUDAToolkit)
find_package(fmt)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

find_package(mathdx REQUIRED
    COMPONENTS curanddx # Puedes añadir más componentes aquí
    CONFIG
    PATHS "${mathdx_ROOT}"
)

add_library(fecuda_core STATIC
    ${CORE_SOURCES}
    ${ALGORITHM_SOURCES}
    ${UTILS_SOURCES}
    ${KERNEL_SOURCES}
    ${KERNEL_MAXMIN_SOURCES}
    ${KERNEL_UTILS_SOURCES}
    ${VALIDATION_SOURCES}
)

set_target_properties(fecuda_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(fecuda_core
    PUBLIC
        CUDA::curand
        mathdx::curanddx
        Python3::Python
        Python3::NumPy
        fmt::fmt
        cudnn
        cutensor
        cublas
        ${CUDA_LIBRARIES}
)
target_compile_definitions(fecuda_core
    PUBLIC
        FECUDA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

target_include_directories(fecuda_core
    PUBLIC
        ${Python3_INCLUDE_DIRS}
        ~/libs/matplotlib-cpp
)

target_compile_options(fecuda_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(fecuda_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(fecuda_main ${MAIN_SOURCES})
target_link_libraries(fecuda_main PRIVATE fecuda_core)
target_compile_definitions(fecuda_main PRIVATE FECUDA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_library(effects_api SHARED ${SERVICE_BINDING_SOURCE})
target_link_libraries(effects_api PRIVATE fecuda_core)
target_compile_definitions(effects_api PRIVATE FECUDA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

else()
message(STATUS "No CUDA compiler found")
endif()

# Alias para compatibilidad con scripts existentes
add_custom_target(tu_ejecutable)
add_dependencies(tu_ejecutable fecuda_main)
