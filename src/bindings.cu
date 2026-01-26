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
    bool consumed = false;

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
        // ⚠️ IMPORTANTE: esta memoria es CUDA
        cudaFree(self->dl_tensor.data);
        delete self;
        printf("Deleted from managed (CUDA)\n");
    }

    py::capsule __dlpack__(py::object stream = py::none())
    {
        if (consumed)
            throw std::runtime_error("DLPack tensor already consumed");

        consumed = true;

        auto *managed = new DLManagedTensor();
        managed->dl_tensor.data = data;
        managed->dl_tensor.device = DLDevice{kDLCUDA, 0};
        managed->dl_tensor.ndim = shape.size();
        managed->dl_tensor.dtype = dtype;
        managed->dl_tensor.shape = shape.data();
        managed->dl_tensor.strides = strides.data();
        managed->dl_tensor.byte_offset = 0;
        managed->manager_ctx = nullptr;
        managed->deleter = &DlpackTensorCuda::deleter;

        return py::capsule(managed, "dltensor");
    }
};

py::tuple maxmin_dlpack(TensorResult<__half> &t1, TensorResult<__half> &t2, float thr, int order)
// py::tuple maxmin_dlpack(py::capsule t_1, py::capsule t_2, __half thr, int order)
{
    // auto t1 = new TensorResult<__half>(t_1);
    // auto t2 = new TensorResult<__half>(t_2);
    __half hthr = __float2half(thr);
    auto results = maxmin<__half>(t1, t2, hthr, order);

    // Por ahora usamos la primera iteración
    auto [d_paths, d_values, h_total_count] = results[0];
    std::cout << "paths finded: " << h_total_count << std::endl;

    __half *h_paths = (__half *)malloc(sizeof(__half) * h_total_count);

    cudaMemcpy(h_paths, d_paths, sizeof(__half) * h_total_count,cudaMemcpyDeviceToHost);


    for (int i = 0; i < h_total_count;i++){
        std::cout <<i <<" value: " << __half2float(h_paths[i]) << std::endl;
    }

        // 🔹 Suponemos h_total_count conocido
        int64_t count = h_total_count;

    auto paths = new DlpackTensorCuda<int4>(
        d_paths,
        {count, 4}, // → (count, 4) por lanes
        int4_dtype()
    );

    auto values = new DlpackTensorCuda<__half>(d_values, {count}, half_dtype());

    return py::make_tuple(paths, values);
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
