cmake_minimum_required(VERSION 3.18)
project(FeCUDA CUDA CXX)

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
include_directories(/usr/include)   

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
file(GLOB TEST_SOURCES "tests/*.cu")
file(GLOB BENCHMARK_SOURCES "benchmarks/*.cu")
file(GLOB EXAMPLE_SOURCES "examples/*.cu")

# Temporalmente excluir kernel_v1_f16.cu hasta que se corrija el tipo de datos
list(REMOVE_ITEM KERNEL_MAXMIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/maxmin/kernel_v1_f16.cu")

# Recopilar todas las fuentes para el ejecutable principal
set(ALL_MAIN_SOURCES 
    ${MAIN_SOURCES} 
    ${CORE_SOURCES} 
    ${ALGORITHM_SOURCES} 
    ${UTILS_SOURCES} 
    ${KERNEL_SOURCES}
    ${KERNEL_MAXMIN_SOURCES}
    ${KERNEL_UTILS_SOURCES}
    ${TEST_SOURCES}
)

# Mostrar qué archivos se están compilando (para debug)
message(STATUS "Archivos principales: ${MAIN_SOURCES}")
message(STATUS "Archivos core: ${CORE_SOURCES}")
message(STATUS "Archivos algorithms: ${ALGORITHM_SOURCES}")
message(STATUS "Archivos utils: ${UTILS_SOURCES}")
message(STATUS "Archivos kernels: ${KERNEL_SOURCES}")

# Ejecutable principal
add_executable(fecuda_main ${ALL_MAIN_SOURCES})

# Configurar propiedades del target principal
set_target_properties(fecuda_main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    
)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
enable_language(CUDA)
find_package(CUDAToolkit)
target_link_libraries(fecuda_main CUDA::curand)
else()
message(STATUS "No CUDA compiler found")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(fecuda_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


# Enlazar el ejecutable principal
target_link_libraries(fecuda_main
    cudnn
    cutensor
    cublas
    ${CUDA_LIBRARIES}
)


target_compile_options(fecuda_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                     --extended-lambda                      
>)


# Alias para compatibilidad con scripts existentes
add_custom_target(tu_ejecutable)
add_dependencies(tu_ejecutable fecuda_main)
