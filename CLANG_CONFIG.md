# Configuración de Clang para FeCUDA

Este proyecto ha sido configurado para usar **Clang** como compilador principal en lugar de GCC.

## Cambios realizados

### 1. CMakeLists.txt
- Agregadas variables para especificar Clang como compilador C/C++:
  ```cmake
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
  ```

### 2. Configuración de VS Code

#### `.vscode/c_cpp_properties.json`
- Actualizado compilador: `/usr/bin/clang++`
- Actualizado IntelliSense: `linux-clang-x64`
- Agregados defines: `__CUDACC__`, `CUDA_ENABLED`
- Actualizada ruta CUDA a versión estándar

#### `.vscode/settings.json`
- Deshabilitado C_Cpp.intelliSenseEngine (usa clangd en su lugar)
- Actualizado compilerPath a Clang
- Configurado formatter predeterminado a Clang
- Agregada configuración especial para archivos CUDA

#### `.vscode/tasks.json`
- **Build with Clang**: Tarea de compilación predeterminada
- **Clean and Build with Clang**: Limpia y recompila

#### `.clangd`
- Mejorada configuración para análisis de CUDA
- Agregadas sugerencias de InlayHints
- Agregada visualización de AKA en hover
- Agregado soporte para diagnósticos a color

## Uso

### Compilar desde VS Code
1. Presiona `Ctrl+Shift+B` para ejecutar la tarea de construcción predeterminada
2. O selecciona "Terminal → Ejecutar tarea" y elige "Build with Clang"

### Limpiar y recompilar
```bash
# Desde la terminal, ejecuta la tarea:
Terminal → Ejecutar tarea → Clean and Build with Clang
```

### O desde la línea de comandos
```bash
cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
cmake --build .
```

## Análisis de errores

Los errores ahora se mostrarán usando clangd:
- Los problemas aparecer en el panel "Problemas"
- Errores y advertencias se destacarán directamente en el editor
- Hover sobre errores para más detalles
- InlayHints mostrarán información de tipos

## Extensiones requeridas

Asegúrate de tener instalada:
- **clangd** (llvm-vs-code-extensions.vscode-clangd)
- **CMake Tools** (ms-vscode.cmake-tools) - recomendado

## Notas importantes

- CUDA sigue siendo compilado con nvcc, solo el código C/C++ usa Clang
- El archivo `.clangd` controla la configuración del compilador clangd
- Para cambios en la configuración de compilación, edita `CMakeLists.txt`
- Para cambios en includes o defines, edita `.clangd`
