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
        CUDA_ERROR(cudaDeviceSynchronize());
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

    bind_attribute_class<float, rxmesh::VertexHandle>(m, "VertexAttributeFloat32");
    bind_attribute_class<double, rxmesh::VertexHandle>(m, "VertexAttributeFloat64");
    bind_attribute_class<int32_t, rxmesh::VertexHandle>(m, "VertexAttributeInt32");
    bind_attribute_class<int8_t, rxmesh::VertexHandle>(m, "VertexAttributeInt8");

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
