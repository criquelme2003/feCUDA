cmake_minimum_required(VERSION 3.18)
project(TuProyecto CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Habilitar compilación separable CUDA globalmente
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Incluir directorios de headers
include_directories(include)
include_directories(/usr/include)   

# Directorios donde buscar bibliotecas compartidas
link_directories(/usr/lib/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/libcutensor/12)

# Archivos fuente del proyecto - especificar explícitamente
file(GLOB CUDA_SOURCES "src/*.cu")
file(GLOB CXX_SOURCES "src/*.cpp")
file(GLOB MAXMIN_SOURCES "src/maxmin/*.cu")

set(ALL_SOURCES ${CUDA_SOURCES} ${CXX_SOURCES} ${MAXMIN_SOURCES})

# Mostrar qué archivos se están compilando (para debug)
message(STATUS "Archivos CUDA encontrados: ${CUDA_SOURCES}")
message(STATUS "Archivos C++ encontrados: ${CXX_SOURCES}")
message(STATUS "Archivos maxmin encontrados: ${MAXMIN_SOURCES}")

# Agregar ejecutable
add_executable(tu_ejecutable ${ALL_SOURCES})

# Configurar propiedades del target
set_target_properties(tu_ejecutable PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Enlazar el ejecutable con cuDNN, cuTENSOR y Python
target_link_libraries(tu_ejecutable
    cudnn
    cutensor
    python3.12
    ${CUDA_LIBRARIES}
)
