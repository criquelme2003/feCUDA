#include "../include/core/types.cuh"
#include "../include/headers.cuh"
#include <dlpack/dlpack.h>
#include <driver_types.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

// g_verbose está definido en maxmin.cu (compilado en ambos targets)

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
        LOG(std::cout << "¡! BINDING Destructor Called" << std::endl);

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

// ─────────────────────────────────────────────────────────────────────────────
// DlpackHolder: wrapper genérico compatible con np.from_dlpack / tf.from_dlpack
// Expone __dlpack__() y __dlpack_device__() a Python.
// Toma ownership del DLManagedTensor* (lo libera en el destructor si no fue
// consumido antes por el runtime de numpy/tf).
// ─────────────────────────────────────────────────────────────────────────────
struct DlpackHolder
{
    DLManagedTensor *managed;
    bool consumed = false;

    explicit DlpackHolder(DLManagedTensor *m) : managed(m) {}

    ~DlpackHolder()
    {
        if (!consumed && managed && managed->deleter)
            managed->deleter(managed);
    }

    py::capsule __dlpack__(py::object /*stream*/ = py::none())
    {
        consumed = true;
        return py::capsule(managed, "dltensor");
    }

    py::object __dlpack_device__() const
    {
        return py::make_tuple(
            (int)managed->dl_tensor.device.device_type,
            (int)managed->dl_tensor.device.device_id);
    }

    // Copia los datos desde GPU a un numpy array (útil para debugging)
    py::object to_numpy() const
    {
        auto &dl = managed->dl_tensor;
        int64_t total = 1;
        for (int i = 0; i < dl.ndim; i++) total *= dl.shape[i];

        bool is_int32 = (dl.dtype.code == kDLInt   && dl.dtype.bits == 32);
        bool is_fp16  = (dl.dtype.code == kDLFloat  && dl.dtype.bits == 16);

        size_t elem_size = dl.dtype.bits / 8;
        size_t nbytes    = total * elem_size;

        std::vector<uint8_t> host(nbytes);
        if (dl.device.device_type == kDLCUDA)
            cudaMemcpy(host.data(), dl.data, nbytes, cudaMemcpyDeviceToHost);
        else
            std::memcpy(host.data(), dl.data, nbytes);

        // Construir shape para numpy
        std::vector<ssize_t> shape(dl.ndim), strides(dl.ndim);
        for (int i = 0; i < dl.ndim; i++) {
            shape[i]   = (ssize_t)dl.shape[i];
            strides[i] = (ssize_t)(dl.strides[i] * elem_size);
        }

        py::str fmt = is_int32 ? py::str("i") : (is_fp16 ? py::str("e") : py::str("f"));

        return py::array(py::buffer_info(
            host.data(), (ssize_t)elem_size, fmt.cast<std::string>(),
            (ssize_t)dl.ndim, shape, strides
        )).attr("copy")();
    }
};

static DlpackHolder* make_int32_holder(int* d_ptr, int64_t rows, int64_t cols)
{
    auto *managed  = new DLManagedTensor();
    auto *shape_   = new int64_t[2]{rows, cols};
    auto *strides_ = new int64_t[2]{cols, 1};

    managed->dl_tensor = {d_ptr, {kDLCUDA, 0}, 2,
                          {kDLInt, 32, 1}, shape_, strides_, 0};
    managed->manager_ctx = nullptr;
    managed->deleter = [](DLManagedTensor *self)
    {
        cudaFree(self->dl_tensor.data);
        delete[] self->dl_tensor.shape;
        delete[] self->dl_tensor.strides;
        delete self;
    };
    return new DlpackHolder(managed);
}

static DlpackHolder* make_half_holder(__half* d_ptr, int64_t count)
{
    auto *managed  = new DLManagedTensor();
    auto *shape_   = new int64_t[1]{count};
    auto *strides_ = new int64_t[1]{1};

    managed->dl_tensor = {d_ptr, {kDLCUDA, 0}, 1,
                          {kDLFloat, 16, 1}, shape_, strides_, 0};
    managed->manager_ctx = nullptr;
    managed->deleter = [](DLManagedTensor *self)
    {
        cudaFree(self->dl_tensor.data);
        delete[] self->dl_tensor.shape;
        delete[] self->dl_tensor.strides;
        delete self;
    };
    return new DlpackHolder(managed);
}

py::tuple maxmin_dlpack(py::object a, py::object b, float thr, int order,
                        bool return_paths = true)
{
    TensorResult<__half> t1(a);
    TensorResult<__half> t2(b);
    __half hthr = __float2half(thr);

    auto results = maxmin(t1, t2, hthr, order, return_paths);
    auto [d_paths, d_values, h_total_count, path_width, effective_order] = results[0];

    int64_t count = (int64_t)h_total_count;

    if (count == 0)
    {
        int    *d_ep; __half *d_ev;
        CHECK_CUDA(cudaMalloc(&d_ep, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_ev, sizeof(__half)));
        return py::make_tuple(
            py::cast(make_int32_holder(d_ep, 0, path_width), py::return_value_policy::take_ownership),
            py::cast(make_half_holder (d_ev, 0),             py::return_value_policy::take_ownership),
            effective_order
        );
    }

    return py::make_tuple(
        py::cast(make_int32_holder(d_paths,  count, path_width), py::return_value_policy::take_ownership),
        py::cast(make_half_holder (d_values, count),             py::return_value_policy::take_ownership),
        effective_order
    );
}

py::tuple maxmin_reduced_dlpack(py::object a, py::object b, float thr, int order,
                                bool return_paths = true, float avg_n = 3.0f)
{
    TensorResult<__half> t1(a);
    TensorResult<__half> t2(b);
    __half hthr = __float2half(thr);

    auto results = maxmin_reduced(t1, t2, hthr, order, return_paths, avg_n);
    auto [d_paths, d_values, h_total_count, path_width, effective_order] = results[0];

    int64_t count = (int64_t)h_total_count;

    if (count == 0)
    {
        int    *d_ep; __half *d_ev;
        CHECK_CUDA(cudaMalloc(&d_ep, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_ev, sizeof(__half)));
        return py::make_tuple(
            py::cast(make_int32_holder(d_ep, 0, path_width), py::return_value_policy::take_ownership),
            py::cast(make_half_holder (d_ev, 0),             py::return_value_policy::take_ownership),
            effective_order
        );
    }

    return py::make_tuple(
        py::cast(make_int32_holder(d_paths,  count, path_width), py::return_value_policy::take_ownership),
        py::cast(make_half_holder (d_values, count),             py::return_value_policy::take_ownership),
        effective_order
    );
}

PYBIND11_MODULE(forgethreads, m)
{
    py::class_<TensorResult<__half>>(m, "TensorResult")
        .def(py::init<py::capsule>())
        .def("__dlpack__",        &TensorResult<__half>::__dlpack__)
        .def("__dlpack_device__", &TensorResult<__half>::__dlpack_device__);

    py::class_<DlpackHolder>(m, "DlpackHolder")
        .def("__dlpack__",        &DlpackHolder::__dlpack__, py::arg("stream") = py::none())
        .def("__dlpack_device__", &DlpackHolder::__dlpack_device__)
        .def("to_numpy",          &DlpackHolder::to_numpy);

    py::class_<DlpackTensorCuda<int4>>(m, "DlpackInt4")
        .def("__dlpack__", &DlpackTensorCuda<int4>::__dlpack__, py::arg("stream") = py::none());

    py::class_<DlpackTensorCuda<__half>>(m, "DlpackFloat")
        .def("__dlpack__", &DlpackTensorCuda<__half>::__dlpack__, py::arg("stream") = py::none());

    m.def("set_verbose", [](bool v) { g_verbose = v; });
    m.def("get_verbose", []()       { return g_verbose; });

    m.def("maxmin", &maxmin_dlpack,
          py::arg("a"), py::arg("b"), py::arg("thr"), py::arg("order"),
          py::arg("return_paths") = true);

    m.def("maxmin_reduced", &maxmin_reduced_dlpack,
          py::arg("a"), py::arg("b"), py::arg("thr"), py::arg("order"),
          py::arg("return_paths") = true, py::arg("avg_n") = 3.0f);
}
