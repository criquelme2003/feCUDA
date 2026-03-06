#include "../include/core/types.cuh"
#include "../include/headers.cuh"
#include <dlpack/dlpack.h>
#include <driver_types.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

DLDataType float_dtype()
{
    return DLDataType{kDLFloat, 32, 1};
}

DLDataType half_dtype()
{
    return DLDataType{kDLBfloat, 16, 1};
}

DLDataType int4_dtype()
{
    return DLDataType{kDLInt, 32, 4}; // 👈 lanes = 4
}

template <typename T> struct DlpackTensorCuda
{
    T *data;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLDataType dtype;

    DlpackTensorCuda(T *ptr, std::vector<int64_t> shape_, DLDataType dtype_)
        : data(ptr), shape(shape_), dtype(dtype_)
    {

        // C-contiguous strides
        strides.resize(shape.size());
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    static void deleter(DLManagedTensor *self)
    {
        std::cout << "¡! BINDING Destructor Called" << std::endl;

        // // ⚠️ IMPORTANTE: esta memoria es CUDA

        // cudaFree(self->dl_tensor.data);
        // delete self;
        // // Comentado porque el free se invoca desde el owner (python)
        // // printf("Deleted from managed (CUDA)\n");

        if (self->dl_tensor.data && self->dl_tensor.device.device_type == kDLCUDA)
        {
            cudaFree(self->dl_tensor.data);
        }
        // Liberar shape si se asignó por separado
        if (self->manager_ctx)
        {
            auto *ctx = static_cast<DlpackTensorCuda *>(self->manager_ctx);
            delete ctx;
        }
        delete self;
    }

    py::capsule __dlpack__(py::object stream = py::none())
    {

        // Asignar shape dinámicamente
        int64_t *shape_copy = new int64_t[shape.size()];
        std::copy(shape.begin(), shape.end(), shape_copy);

        int64_t *strides_copy = new int64_t[strides.size()];
        std::copy(strides.begin(), strides.end(), strides_copy);

        auto *managed = new DLManagedTensor();
        managed->dl_tensor.data = data;
        managed->dl_tensor.device = DLDevice{kDLCUDA, 0};
        managed->dl_tensor.ndim = shape.size();
        managed->dl_tensor.dtype = dtype;
        managed->dl_tensor.shape = shape_copy;     // ✅ Copia independiente
        managed->dl_tensor.strides = strides_copy; // ✅ Copia independiente
        managed->dl_tensor.byte_offset = 0;
        managed->manager_ctx = this;

        managed->deleter = [](DLManagedTensor *self)
        {
            if (!self)
                return;
            if (self->dl_tensor.device.device_type == kDLCUDA)
            {
                cudaFree(self->dl_tensor.data);
            }
            delete[] self->dl_tensor.shape;
            delete[] self->dl_tensor.strides;
            delete self;
        };

        return py::capsule(managed, "dltensor");
    }
};

py::tuple maxmin_dlpack(py::object a, py::object b, float thr, int order)
{
    // 🔹 Convertir automáticamente usando __dlpack__()
    TensorResult<__half> t1(a);
    std::cout << "tensor 1 creado correctamente"  << std::endl;

    TensorResult<__half> t2(b);

    std::cout << "tensor 2 creado correctamente"  << std::endl;

    __half hthr = __float2half(thr);


    // Ejecutar tu kernel
    auto results = maxmin(t1, t2, hthr, order);

    // Tomamos primera iteración
    auto [d_paths, d_values, h_total_count] = results[0];

    std::cout << "paths finded: " << h_total_count << std::endl;

    int64_t count = h_total_count;

    // 🔹 Crear tensores resultado
    auto paths = new TensorResult<int4>(
        MemorySpace::Device,
        count,
        4,
        1,
        1
    );

    auto values = new TensorResult<__half>(
        MemorySpace::Device,
        count,
        1,
        1,
        1
    );

    // copiar resultados del kernel
    CHECK_CUDA(cudaMemcpy(
        paths->getData(),
        d_paths,
        count * sizeof(int4),
        cudaMemcpyDeviceToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        values->getData(),
        d_values,
        count * sizeof(__half),
        cudaMemcpyDeviceToDevice
    ));

    CHECK_CUDA(cudaDeviceSynchronize());

    // 🔹 devolver como DLPack
    return py::make_tuple(
        paths->__dlpack__(),
        values->__dlpack__()
    );
}

PYBIND11_MODULE(forgethreads, m)
{
    py::class_<TensorResult<__half>>(m, "TensorResult")
        .def(py::init<py::capsule>()) // 👈 NUEVO
        .def("__dlpack__", &TensorResult<__half>::__dlpack__)
        .def("__dlpack_device__", &TensorResult<__half>::__dlpack_device__);

    py::class_<DlpackTensorCuda<int4>>(m, "DlpackInt4")
        .def("__dlpack__", &DlpackTensorCuda<int4>::__dlpack__, py::arg("stream") = py::none());

    py::class_<DlpackTensorCuda<__half>>(m, "DlpackFloat")
        .def("__dlpack__", &DlpackTensorCuda<__half>::__dlpack__, py::arg("stream") = py::none());

    m.def("maxmin", &maxmin_dlpack);
}
