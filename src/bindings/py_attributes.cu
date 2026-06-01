#include "bindings/dlpack_minimal.h"
#include "bindings/py_attribute.h"

#include <cuda_runtime_api.h>

using namespace rxmesh;

namespace pyrxmesh_py {

namespace {

struct AttributeDlpackContext
{
    std::shared_ptr<PyAttributeBase> owner;
    int64_t                          shape[2];
    int64_t                          strides[2];
};

template <typename T>
__global__ void copy_dlpack_to_soa_attribute_kernel(T*       dst,
                                                    const T* src,
                                                    int64_t  rows,
                                                    int64_t  cols,
                                                    int64_t  stride0,
                                                    int64_t  stride1)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n   = rows * cols;
    if (idx >= n) {
        return;
    }

    const int64_t row     = idx % rows;
    const int64_t col     = idx / rows;
    dst[col * rows + row] = src[row * stride0 + col * stride1];
}

template <typename T>
__global__ void copy_dlpack_to_row_major_kernel(T*       dst,
                                                const T* src,
                                                int64_t  rows,
                                                int64_t  cols,
                                                int64_t  stride0,
                                                int64_t  stride1)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n   = rows * cols;
    if (idx >= n) {
        return;
    }

    const int64_t row     = idx / cols;
    const int64_t col     = idx % cols;
    dst[row * cols + col] = src[row * stride0 + col * stride1];
}

dlpack::DLDataType attribute_dlpack_dtype(DType dtype)
{
    switch (dtype) {
        case DType::Float32:
            return {2, 32, 1};
        case DType::Float64:
            return {2, 64, 1};
        case DType::Int32:
            return {0, 32, 1};
        case DType::Int8:
            return {0, 8, 1};
        default:
            throw std::invalid_argument(
                "Attribute.to_dlpack() encountered an unsupported dtype.");
    }
}

bool dlpack_dtype_equal(dlpack::DLDataType lhs, dlpack::DLDataType rhs)
{
    return lhs.code == rhs.code && lhs.bits == rhs.bits &&
           lhs.lanes == rhs.lanes;
}

void require_attribute_tensor_view(const PyAttributeBase& self,
                                   rxmesh::locationT      location,
                                   const char*            api)
{
    if (location != rxmesh::HOST && location != rxmesh::DEVICE) {
        throw std::invalid_argument(
            std::string(api) +
            " location must be Location.HOST or  Location.DEVICE.");
    }
    if (!self.is_tensor_layout()) {
        throw std::runtime_error(
            std::string(api) +
            " only supports zero-copy views for Layout.SoA attributes. Use "
            "the explicit *_copy API for AoS/AoSoA attributes.");
    }
    if (location == rxmesh::HOST && !self.is_host_allocated()) {
        throw std::runtime_error(
            std::string(api) +
            " requires an existing HOST allocation. Move or create the "
            "attribute on HOST, or use the explicit *_copy API.");
    }
    if (location == rxmesh::DEVICE && !self.is_device_allocated()) {
        throw std::runtime_error(
            std::string(api) +
            " requires an existing DEVICE allocation. Move or create the "
            "attribute on DEVICE, or use the explicit *_copy API.");
    }
}

void dlpack_capsule_destructor(PyObject* capsule)
{
    if (PyCapsule_IsValid(capsule, "used_dltensor")) {
        return;
    }
    if (!PyCapsule_IsValid(capsule, "dltensor")) {
        return;
    }
    auto* managed = static_cast<dlpack::DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed && managed->deleter) {
        managed->deleter(managed);
    }
}

void dlpack_managed_tensor_deleter(dlpack::DLManagedTensor* self)
{
    delete static_cast<AttributeDlpackContext*>(self->manager_ctx);
    delete self;
}

py::capsule attribute_to_dlpack(std::shared_ptr<PyAttributeBase> self,
                                int                              location,
                                py::object)
{
    const auto loc = parse_location(location);
    require_attribute_tensor_view(*self, loc, "Attribute.to_dlpack()");

    if (loc == rxmesh::DEVICE) {
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
    }

    auto* managed  = new dlpack::DLManagedTensor();
    auto* context  = new AttributeDlpackContext();
    context->owner = std::move(self);

    context->shape[0]   = context->owner->element_count();
    context->shape[1]   = context->owner->dim();
    context->strides[0] = 1;
    context->strides[1] = context->owner->element_count();

    int device_id = 0;
    if (loc == rxmesh::DEVICE) {
        CUDA_ERROR(cudaGetDevice(&device_id));
    }

    managed->dl_tensor.data   = context->owner->data_ptr(loc);
    managed->dl_tensor.device = {
        loc == rxmesh::DEVICE ? dlpack::kDLCUDA : dlpack::kDLCPU, device_id};
    managed->dl_tensor.ndim  = 2;
    managed->dl_tensor.dtype = attribute_dlpack_dtype(context->owner->dtype());
    managed->dl_tensor.shape = context->shape;
    managed->dl_tensor.strides     = context->strides;
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx           = context;
    managed->deleter               = dlpack_managed_tensor_deleter;

    return py::capsule(managed, "dltensor", dlpack_capsule_destructor);
}

py::capsule attribute_dunder_dlpack(std::shared_ptr<PyAttributeBase> self,
                                    py::object                       stream)
{
    const int loc = self->is_device_allocated() ?
                        static_cast<int>(rxmesh::DEVICE) :
                        static_cast<int>(rxmesh::HOST);
    return attribute_to_dlpack(std::move(self), loc, std::move(stream));
}

py::tuple attribute_dlpack_device(const PyAttributeBase& self)
{
    if (self.is_device_allocated()) {
        int device_id = 0;
        CUDA_ERROR(cudaGetDevice(&device_id));
        return py::make_tuple(static_cast<int>(dlpack::kDLCUDA), device_id);
    }
    if (self.is_host_allocated()) {
        return py::make_tuple(static_cast<int>(dlpack::kDLCPU), 0);
    }
    throw std::runtime_error("Attribute has no allocated memory.");
}

py::object acquire_dlpack_capsule(py::object source)
{
    if (PyCapsule_IsValid(source.ptr(), "dltensor")) {
        return source;
    }
    if (!py::hasattr(source, "__dlpack__")) {
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() expects a DLPack capsule or an "
            "object with __dlpack__().");
    }
    return source.attr("__dlpack__")();
}

void mark_dlpack_capsule_consumed(py::object               capsule,
                                  dlpack::DLManagedTensor* managed)
{
    if (managed && managed->deleter) {
        managed->deleter(managed);
    }
    PyCapsule_SetName(capsule.ptr(), "used_dltensor");
    PyCapsule_SetDestructor(capsule.ptr(), nullptr);
}

void validate_dlpack_attribute_shape(const PyAttributeBase&  self,
                                     const dlpack::DLTensor& tensor)
{
    if (tensor.ndim != 1 && tensor.ndim != 2) {
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() expects a 1D or 2D tensor.");
    }

    if (tensor.shape[0] != static_cast<int64_t>(self.element_count())) {
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() row count does not match the mesh "
            "element count.");
    }

    if (tensor.ndim == 1 && self.dim() != 1) {
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() received a 1D tensor for a "
            "multi-column attribute.");
    }

    if (tensor.ndim == 2 &&
        tensor.shape[1] != static_cast<int64_t>(self.dim())) {
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() column count does not match the "
            "attribute dimension.");
    }
}

template <typename T, typename HandleT>
void copy_host_flat_to_attribute(rxmesh::Attribute<T, HandleT>& attr,
                                 const T*                       src,
                                 int64_t                        stride0,
                                 int64_t                        stride1)
{
    for (uint32_t i = 0; i < attr.rows(); ++i) {
        for (uint32_t j = 0; j < attr.cols(); ++j) {
            attr(i, j) = src[static_cast<int64_t>(i) * stride0 +
                             static_cast<int64_t>(j) * stride1];
        }
    }
}

template <typename T>
void copy_cuda_dlpack_to_host_flat(const dlpack::DLTensor& tensor,
                                   int64_t                 rows,
                                   int64_t                 cols,
                                   int64_t                 stride0,
                                   int64_t                 stride1,
                                   std::vector<T>&         host)
{
    const auto* src = reinterpret_cast<const T*>(
        static_cast<const char*>(tensor.data) + tensor.byte_offset);
    const int64_t n = rows * cols;

    T* d_compact = nullptr;
    CUDA_ERROR(cudaMalloc(&d_compact, static_cast<size_t>(n) * sizeof(T)));

    constexpr int threads = 256;
    copy_dlpack_to_row_major_kernel<T>
        <<<static_cast<int>((n + threads - 1) / threads), threads>>>(
            d_compact, src, rows, cols, stride0, stride1);
    CUDA_ERROR(cudaGetLastError());

    host.resize(static_cast<size_t>(n));

    CUDA_ERROR(cudaMemcpy(host.data(),
                          d_compact,
                          static_cast<size_t>(n) * sizeof(T),
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaStreamSynchronize(nullptr));
    CUDA_ERROR(cudaFree(d_compact));
}

template <typename T, typename HandleT>
void copy_dlpack_to_attribute(PyAttribute<T, HandleT>& self,
                              const dlpack::DLTensor&  tensor,
                              rxmesh::locationT        target)
{
    validate_dlpack_attribute_shape(self, tensor);

    const auto expected_dtype = attribute_dlpack_dtype(self.dtype());
    if (!dlpack_dtype_equal(tensor.dtype, expected_dtype)) {
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() tensor dtype does not match the "
            "attribute dtype.");
    }

    if ((target & (rxmesh::HOST | rxmesh::DEVICE)) == rxmesh::LOCATION_NONE) {
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() target must include Location.HOST "
            "or Location.DEVICE.");
    }

    const int64_t rows = static_cast<int64_t>(self.element_count());
    const int64_t cols = static_cast<int64_t>(self.dim());
    const int64_t stride0 =
        tensor.strides ? tensor.strides[0] : (tensor.ndim == 1 ? 1 : cols);
    const int64_t stride1 =
        tensor.ndim == 1 ? 1 : (tensor.strides ? tensor.strides[1] : 1);

    if (tensor.device.device_type == dlpack::kDLCPU) {
        const auto* src = reinterpret_cast<const T*>(
            static_cast<const char*>(tensor.data) + tensor.byte_offset);
        ensure_host_writable(*self.attr);
        copy_host_flat_to_attribute(*self.attr, src, stride0, stride1);
        if ((target & rxmesh::DEVICE) == rxmesh::DEVICE) {
            self.attr->move(rxmesh::HOST, rxmesh::DEVICE);
        }
        return;
    }

    if (tensor.device.device_type != dlpack::kDLCUDA) {
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() supports CPU and CUDA DLPack "
            "tensors.");
    }

    if ((target & rxmesh::DEVICE) == rxmesh::DEVICE &&
        self.attr->is_tensor_layout()) {
        ensure_allocated(*self.attr, rxmesh::DEVICE);

        const auto* src = reinterpret_cast<const T*>(
            static_cast<const char*>(tensor.data) + tensor.byte_offset);
        constexpr int threads = 256;
        const int64_t n       = rows * cols;
        copy_dlpack_to_soa_attribute_kernel<T>
            <<<static_cast<int>((n + threads - 1) / threads), threads>>>(
                self.attr->data(rxmesh::DEVICE),
                src,
                rows,
                cols,
                stride0,
                stride1);
        CUDA_ERROR(cudaGetLastError());
        CUDA_ERROR(cudaStreamSynchronize(nullptr));

        if ((target & rxmesh::HOST) == rxmesh::HOST) {
            self.attr->move(rxmesh::DEVICE, rxmesh::HOST);
        }
        return;
    }
    std::vector<T> host(static_cast<size_t>(rows * cols));
    copy_cuda_dlpack_to_host_flat<T>(
        tensor, rows, cols, stride0, stride1, host);
    ensure_host_writable(*self.attr);
    copy_host_flat_to_attribute(*self.attr, host.data(), cols, 1);
    if ((target & rxmesh::DEVICE) == rxmesh::DEVICE) {
        self.attr->move(rxmesh::HOST, rxmesh::DEVICE);
    }
}

template <typename T, typename HandleT>
bool try_copy_dlpack_to_attribute(PyAttributeBase&        self,
                                  const dlpack::DLTensor& tensor,
                                  rxmesh::locationT       target)
{
    auto* typed = dynamic_cast<PyAttribute<T, HandleT>*>(&self);
    if (!typed) {
        return false;
    }
    copy_dlpack_to_attribute(*typed, tensor, target);
    return true;
}

void attribute_from_dlpack_copy(PyAttributeBase& self,
                                py::object       source,
                                int              target)
{
    py::object capsule = acquire_dlpack_capsule(std::move(source));

    auto* managed = static_cast<dlpack::DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor"));

    if (!managed) {
        PyErr_Clear();
        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() received an invalid DLPack "
            "capsule.");
    }

    const auto dst = parse_location(target);

    try {
        const dlpack::DLTensor& tensor = managed->dl_tensor;
        if (try_copy_dlpack_to_attribute<float, rxmesh::VertexHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<double, rxmesh::VertexHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<int32_t, rxmesh::VertexHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<int8_t, rxmesh::VertexHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<float, rxmesh::EdgeHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<double, rxmesh::EdgeHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<int32_t, rxmesh::EdgeHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<int8_t, rxmesh::EdgeHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<float, rxmesh::FaceHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<double, rxmesh::FaceHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<int32_t, rxmesh::FaceHandle>(
                self, tensor, dst) ||
            try_copy_dlpack_to_attribute<int8_t, rxmesh::FaceHandle>(
                self, tensor, dst)) {
            mark_dlpack_capsule_consumed(capsule, managed);
            return;
        }

        throw std::invalid_argument(
            "Attribute.from_dlpack_copy() received an unsupported attribute "
            "type.");
    } catch (...) {
        mark_dlpack_capsule_consumed(capsule, managed);
        throw;
    }
}

template <typename T, typename HandleT>
void bind_attribute_class(py::module_& m, const char* class_name)
{
    py::class_<PyAttribute<T, HandleT>,
               PyAttributeBase,
               std::shared_ptr<PyAttribute<T, HandleT>>>(m, class_name);
}

}  // namespace

void register_attribute(py::module_& m)
{
    py::class_<PyAttributeBase, std::shared_ptr<PyAttributeBase>>(m,
                                                                  "Attribute")
        .def_property_readonly("name", &PyAttributeBase::name)
        .def_property_readonly("dtype",
                               [](const PyAttributeBase& self) {
                                   return std::string(dtype_name(self.dtype()));
                               })
        .def_property_readonly(
            "element_kind",
            [](const PyAttributeBase& self) {
                return std::string(element_kind_name(self.element_kind()));
            })
        .def_property_readonly("dim", &PyAttributeBase::dim)
        .def_property_readonly("size", &PyAttributeBase::size)
        .def_property_readonly("element_count", &PyAttributeBase::element_count)
        .def_property_readonly("shape", &PyAttributeBase::shape)
        .def_property_readonly("bytes", &PyAttributeBase::bytes)
        .def_property_readonly("allocated", &PyAttributeBase::allocated)
        .def_property_readonly("layout", &PyAttributeBase::layout)
        .def_property_readonly("is_host_allocated",
                               &PyAttributeBase::is_host_allocated)
        .def_property_readonly("is_device_allocated",
                               &PyAttributeBase::is_device_allocated)
        .def_property_readonly("is_tensor_layout",
                               &PyAttributeBase::is_tensor_layout)
        .def("reset",
             &PyAttributeBase::reset,
             py::arg("value"),
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL))
        .def("move",
             &PyAttributeBase::move,
             py::arg("source"),
             py::arg("target"))
        .def("copy_from",
             &PyAttributeBase::copy_from,
             py::arg("other"),
             py::arg("source") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL))
        .def(
            "to_numpy",
            [](std::shared_ptr<PyAttributeBase> self, int location) {
                return self->to_numpy(location, py::cast(self));
            },
            py::arg("location") = static_cast<int>(rxmesh::HOST),
            "Return a zero-copy NumPy view of true Layout.SoA HOST memory.")
        .def("to_numpy_copy",
             &PyAttributeBase::to_numpy_copy,
             py::arg("source") = static_cast<int>(rxmesh::HOST),
             "Copy this attribute to a NumPy array in input/global element "
             "order.")
        .def("from_numpy_copy",
             &PyAttributeBase::from_numpy_copy,
             py::arg("values"),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
             "Copy a NumPy-compatible array into RXMesh attribute memory.")
        .def("from_dlpack_copy",
             &attribute_from_dlpack_copy,
             py::arg("source"),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
             "Copy a CPU or CUDA DLPack tensor into RXMesh attribute memory.")
        .def("to_matrix_copy",
             &PyAttributeBase::to_matrix_copy,
             "Copy this attribute into a DenseMatrix using RXMesh handle "
             "order.")
        .def("from_matrix_copy",
             &PyAttributeBase::from_matrix_copy,
             py::arg("matrix"),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
             "Copy a dense matrix into RXMesh attribute memory.")
        .def(
            "to_dlpack",
            [](std::shared_ptr<PyAttributeBase> self,
               int                              location,
               py::object                       stream) {
                return attribute_to_dlpack(
                    std::move(self), location, std::move(stream));
            },
            py::arg("location") = static_cast<int>(rxmesh::DEVICE),
            py::arg("stream")   = py::none(),
            "Return a DLPack capsule that views true Layout.SoA RXMesh memory.")
        .def(
            "__dlpack__",
            [](std::shared_ptr<PyAttributeBase> self, py::object stream) {
                return attribute_dunder_dlpack(std::move(self),
                                               std::move(stream));
            },
            py::arg("stream") = py::none())
        .def("__dlpack_device__",
             [](const PyAttributeBase& self) {
                 return attribute_dlpack_device(self);
             })
        .def("norm2", &PyAttributeBase::norm2, py::arg("column") = py::none())
        .def("dot",
             &PyAttributeBase::dot,
             py::arg("other"),
             py::arg("column") = py::none())
        .def("reduce_sum",
             &PyAttributeBase::reduce_sum,
             py::arg("column") = py::none())
        .def("reduce_min",
             &PyAttributeBase::reduce_min,
             py::arg("column") = py::none())
        .def("reduce_max",
             &PyAttributeBase::reduce_max,
             py::arg("column") = py::none())
        .def("argmax",
             &PyAttributeBase::argmax,
             py::arg("column") = py::none(),
             "Return (handle, value) for the maximum attribute value.")
        .def("argmin",
             &PyAttributeBase::argmin,
             py::arg("column") = py::none(),
             "Return (handle, value) for the minimum attribute value.")
        .def("__rxmesh_capsule__",
             &PyAttributeBase::capsule,
             "Return a low-level capsule for compiled PyRXMesh plugins.");

    bind_attribute_class<float, rxmesh::VertexHandle>(m,
                                                      "VertexAttributeFloat32");
    bind_attribute_class<double, rxmesh::VertexHandle>(
        m, "VertexAttributeFloat64");
    bind_attribute_class<int32_t, rxmesh::VertexHandle>(m,
                                                        "VertexAttributeInt32");
    bind_attribute_class<int8_t, rxmesh::VertexHandle>(m,
                                                       "VertexAttributeInt8");

    bind_attribute_class<float, rxmesh::EdgeHandle>(m, "EdgeAttributeFloat32");
    bind_attribute_class<double, rxmesh::EdgeHandle>(m, "EdgeAttributeFloat64");
    bind_attribute_class<int32_t, rxmesh::EdgeHandle>(m, "EdgeAttributeInt32");
    bind_attribute_class<int8_t, rxmesh::EdgeHandle>(m, "EdgeAttributeInt8");

    bind_attribute_class<float, rxmesh::FaceHandle>(m, "FaceAttributeFloat32");
    bind_attribute_class<double, rxmesh::FaceHandle>(m, "FaceAttributeFloat64");
    bind_attribute_class<int32_t, rxmesh::FaceHandle>(m, "FaceAttributeInt32");
    bind_attribute_class<int8_t, rxmesh::FaceHandle>(m, "FaceAttributeInt8");
}

}  // namespace pyrxmesh_py
