# Guía Rcpp para exponer `forgethreads` a R

Esta nota resume lo necesario para construir un wrapper Rcpp sobre `forgethreads` siguiendo el diseño acordado:

- `TensorResult` debe quedar como clase **core limpia**
- el wrapper de R debe **recibir y devolver tipos nativos de R**
- la lógica CUDA debe permanecer fuera de la frontera R

La idea no es exponer objetos Python ni DLPack en R. La frontera correcta es:

`R nativo -> Rcpp wrapper -> core C++/CUDA -> Rcpp wrapper -> R nativo`

---

## 1. Principio de diseño

### Lo que sí debe vivir en el core

- shape
- puntero a datos
- ownership de memoria
- memoria host/device
- copias host <-> device
- helpers para reservar/liberar
- llamada a kernels CUDA

### Lo que no debe vivir en el core

- `py::object`
- `pybind11`
- `SEXP`
- `Rcpp::NumericVector`
- lógica específica de DLPack
- lógica específica de R

En otras palabras:

- `TensorResult` debe ser una clase C++/CUDA pura
- Python debe tener su adapter
- R debe tener su adapter

---

## 2. Qué tipos usar en el wrapper Rcpp

Para la frontera con R, usa tipos que Rcpp entiende de forma natural.

### Entradas recomendadas

- `Rcpp::NumericVector`
- `Rcpp::IntegerVector`
- `Rcpp::NumericMatrix`
- `Rcpp::IntegerMatrix`
- `Rcpp::LogicalVector`

Si tus tensores vienen como arrays 3D desde R, normalmente entrarán como:

- `NumericVector` con atributo `dim`

Eso es lo más común y más flexible.

### Salidas recomendadas

- `Rcpp::List`
- `Rcpp::NumericVector`
- `Rcpp::IntegerVector`
- `Rcpp::NumericMatrix`
- `Rcpp::IntegerMatrix`

Para `maxmin`, la salida más razonable suele ser un `List`, por ejemplo:

```cpp
return Rcpp::List::create(
    Rcpp::Named("paths") = paths_matrix,
    Rcpp::Named("values") = values_vector,
    Rcpp::Named("effective_order") = effective_order
);
```

### Lo que no conviene exponer a R

- `TensorResult`
- punteros crudos
- `DLManagedTensor`
- clases CUDA
- `std::vector<__half>`

R no sabe manejar eso directamente. Si quieres usar esas clases, hazlo solo dentro del wrapper.

---

## 3. Atributos y etiquetas de Rcpp que sí importan

### Exportar una función a R

```cpp
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List maxmin_gpu_cpp(Rcpp::NumericVector a,
                          Rcpp::NumericVector b,
                          double thr,
                          int order);
```

Eso le dice a Rcpp que genere el código de registro para llamar la función desde R.

### Incluir dependencias

Si usas compilación con `sourceCpp`, a veces verás:

```cpp
// [[Rcpp::depends(Rcpp)]]
```

En un paquete R formal normalmente esto se maneja en:

- `DESCRIPTION`
- `NAMESPACE`
- `src/Makevars`

No es la pieza principal del diseño, pero sí conviene conocerla.

### Excepciones

Si lanzas `std::runtime_error`, Rcpp la traduce bastante bien a error de R:

```cpp
throw std::runtime_error("Input tensors must have rank 3");
```

Eso está bien y es preferible a `exit()` dentro del wrapper de R.

---

## 4. Cómo debería verse `TensorResult` limpio

La versión actual mezcla responsabilidades del core con Python. Para R conviene separarla.

### Objetivo

`TensorResult` debe representar un tensor o buffer del core, no un objeto de Python ni de R.

### Responsabilidades correctas

- guardar `data`
- guardar `dims`
- saber si vive en host o device
- reservar/liberar memoria
- copiar host <-> device
- exponer getters simples

### Responsabilidades incorrectas

- constructor desde `py::object`
- `__dlpack__`
- `__dlpack_device__`
- `py::capsule`
- cualquier lógica Rcpp

### Boceto conceptual

```cpp
enum class MemorySpace {
    Host,
    Device
};

struct TensorDims {
    int b = 0;
    int m = 0;
    int n = 0;
    int k = 1;
};

template <typename T>
class TensorResult {
public:
    TensorResult(MemorySpace space, TensorDims dims);
    ~TensorResult();

    T* data();
    const T* data() const;

    TensorDims dims() const;
    size_t size_bytes() const;

    void move_to_device();
    void move_to_host();

private:
    T* data_ = nullptr;
    TensorDims dims_;
    MemorySpace space_ = MemorySpace::Host;
};
```

El adapter Python puede convertir `py::object -> TensorResult<__half>`.

El adapter R puede convertir `Rcpp::NumericVector -> TensorResult<float>`.

---

## 5. Qué debe hacer el wrapper Rcpp

El wrapper Rcpp no debe contener la lógica del algoritmo. Debe hacer solo estas tareas:

1. validar inputs de R
2. leer `dim`
3. convertir a formato del core
4. llamar al core CUDA
5. traer resultados a host
6. devolver estructuras nativas de R

### Ejemplo de firma recomendable

```cpp
// [[Rcpp::export]]
Rcpp::List maxmin_gpu_cpp(Rcpp::NumericVector a,
                          Rcpp::NumericVector b,
                          double thr,
                          int order,
                          bool return_paths = true);
```

### Validaciones mínimas

- `a` y `b` deben tener atributo `dim`
- deben ser rank 3
- sus shapes deben coincidir
- `order >= 1`
- `thr` en rango válido para el algoritmo

### Conversión sugerida

R usa `double` por defecto.

Mi recomendación para una primera versión:

- recibir `NumericVector` en `double`
- convertir a `float` o `__half` dentro del wrapper/core

Eso simplifica mucho la interfaz con R.

---

## 6. Ejemplo de wrapper Rcpp

Este ejemplo muestra la forma, no el detalle exacto del algoritmo.

```cpp
#include <Rcpp.h>
#include "forgethreads_core.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
List maxmin_gpu_cpp(NumericVector a,
                    NumericVector b,
                    double thr,
                    int order,
                    bool return_paths = true) {

    IntegerVector dim_a = a.attr("dim");
    IntegerVector dim_b = b.attr("dim");

    if (dim_a.size() != 3 || dim_b.size() != 3) {
        stop("a and b must be rank-3 arrays");
    }

    if (dim_a[0] != dim_b[0] || dim_a[1] != dim_b[1] || dim_a[2] != dim_b[2]) {
        stop("a and b must have the same shape");
    }

    TensorDims dims{dim_a[0], dim_a[1], dim_a[2], 1};

    TensorResult<float> t1(MemorySpace::Host, dims);
    TensorResult<float> t2(MemorySpace::Host, dims);

    std::copy(a.begin(), a.end(), t1.data());
    std::copy(b.begin(), b.end(), t2.data());

    auto result = maxmin_host_wrapper(t1, t2, static_cast<float>(thr), order, return_paths);

    IntegerMatrix paths = result.paths;
    NumericVector values = result.values;

    return List::create(
        Named("paths") = paths,
        Named("values") = values,
        Named("effective_order") = result.effective_order
    );
}
```

El punto importante es este:

- el wrapper recibe `NumericVector`
- internamente usa `TensorResult`
- devuelve `List`, `IntegerMatrix`, `NumericVector`

Eso es exactamente el patrón que conviene.

---

## 7. Detección de GPU y fallback

Como la idea es usar tu backend solo cuando haya GPU, conviene exponer una función pequeña:

```cpp
// [[Rcpp::export]]
bool has_gpu_cpp() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}
```

Y luego en R:

```r
maxmin_auto <- function(a, b, thr, order, return_paths = TRUE) {
  if (has_gpu_cpp()) {
    maxmin_gpu_cpp(a, b, thr, order, return_paths)
  } else {
    maxmin_cpu_r(a, b, thr, order, return_paths)
  }
}
```

Ese dispatcher te deja reemplazar el algoritmo actual sin cambiar demasiado la API del usuario.

---

## 8. Qué no haría

- no expondría DLPack a R
- no intentaría que R conozca `TensorResult`
- no mezclaría `Rcpp` y `pybind11` dentro del mismo wrapper
- no devolvería memoria GPU cruda a R
- no haría que el wrapper R contenga la lógica del kernel

Todo eso complica la frontera y te deja un diseño más frágil.

---

## 9. Estructura sugerida del código

Una división sana sería:

```text
include/
  core/
    tensor_result.hpp        <- clase limpia
    maxmin_core.hpp          <- API del core

src/
  algorithms/
    maxmin.cu                <- lógica CUDA core
  bindings/
    python_bindings.cu       <- pybind11 + DLPack
    r_bindings.cpp           <- Rcpp + SEXP/List/NumericVector
```

### Regla de oro

- `core/` no conoce Python ni R
- `python_bindings.cu` conoce Python
- `r_bindings.cpp` conoce R

---

## 10. Checklist mínimo antes de escribir el wrapper Rcpp

- separar `TensorResult` del constructor `py::object`
- mover lógica DLPack fuera del core
- definir una API C++ pura para `maxmin`
- decidir formato de salida para R
- implementar `has_gpu_cpp()`
- implementar `maxmin_gpu_cpp()`
- crear `maxmin_auto()` del lado R

---

## 11. Recomendación final

Si quieres un wrapper R mantenible, piensa Rcpp como una **capa de traducción**, no como el lugar donde vive el algoritmo.

La combinación correcta es:

- `TensorResult` limpio en el core
- wrapper Rcpp con tipos nativos
- fallback a la implementación R cuando no haya GPU

Ese camino te deja:

- una API R simple
- un core reusable
- compatibilidad con Python y R sin duplicar demasiado código
