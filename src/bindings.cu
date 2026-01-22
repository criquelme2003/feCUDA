#include <pybind11/pybind11.h>
#include "../include/core/types.cuh"

namespace py = pybind11;

PYBIND11_MODULE(forgethreads, m)
{
  py::class_<TensorResult<float>>(m, "TensorResult")
      .def(py::init<MemorySpace, int, int, int, int>(),
           py::arg("space"),
           py::arg("b"),
           py::arg("m"),
           py::arg("n"),
           py::arg("k") = 1)
      .def("__dlpack__", &TensorResult<float>::__dlpack__)
      .def("__dlpack_device__", &TensorResult<float>::__dlpack_device__);

  m.def("getTensor", []()
        {

    // Tensor 1 x 1 x 1 x 6
    auto *t = new TensorResult<float>(MemorySpace::Host, 1, 1, 1, 6);

    float *ptr = t->getData();
    for (int i = 0; i < 6; ++i)
      ptr[i] = i;

    std::cout << "ptr = " << (void *)ptr << std::endl;
    std::cout << "DATA[0] from lambda" << t->getData()[0] << std::endl;

    // ⚠️ Se devuelve por valor → ownership queda en Python
    return t; }, py::return_value_policy::take_ownership);
  
  
}
