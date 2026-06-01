#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime_api.h>

#include "pyrxmesh/plugin_api.h"
#include "rxmesh/handle.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/types.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"

#if USE_POLYSCOPE
#include "polyscope/polyscope.h"
#endif

namespace py = pybind11;

namespace pyrxmesh_py {

inline constexpr uint32_t    plugin_abi_version = PYRXMESH_PLUGIN_ABI_VERSION;
inline constexpr const char* build_config_tag   = PYRXMESH_BUILD_CONFIG;

enum class ElementKind : uint32_t
{
    Vertex = static_cast<uint32_t>(pyrxmesh::ElementKind::Vertex),
    Edge   = static_cast<uint32_t>(pyrxmesh::ElementKind::Edge),
    Face   = static_cast<uint32_t>(pyrxmesh::ElementKind::Face),
};

enum class DType : uint32_t
{
    Float32 = static_cast<uint32_t>(pyrxmesh::DType::Float32),
    Float64 = static_cast<uint32_t>(pyrxmesh::DType::Float64),
    Int8    = static_cast<uint32_t>(pyrxmesh::DType::Int8),
    UInt8   = static_cast<uint32_t>(pyrxmesh::DType::UInt8),
    Int16   = static_cast<uint32_t>(pyrxmesh::DType::Int16),
    UInt16  = static_cast<uint32_t>(pyrxmesh::DType::UInt16),
    Int32   = static_cast<uint32_t>(pyrxmesh::DType::Int32),
    UInt32  = static_cast<uint32_t>(pyrxmesh::DType::UInt32),
    Int64   = static_cast<uint32_t>(pyrxmesh::DType::Int64),
    UInt64  = static_cast<uint32_t>(pyrxmesh::DType::UInt64),

};


template <typename>
struct always_false : std::false_type
{
};


template <typename T>
DType dtype_for()
{
    if constexpr (std::is_same_v<T, float>) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::Float64;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return DType::Int8;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return DType::UInt8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return DType::Int16;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return DType::UInt16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DType::Int32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DType::UInt32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DType::Int64;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return DType::UInt64;
    } else {
        static_assert(always_false<T>::value, "Unsupported PyRXMesh dtype");
    }
}

template <typename T>
const char* dense_dtype_name()
{
    if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, double>) {
        return "float64";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32";
    } else {
        static_assert(always_false<T>::value, "Unsupported DenseMatrix dtype");
    }
}

template <typename HandleT>
ElementKind element_kind_for()
{
    using namespace rxmesh;

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return ElementKind::Vertex;
    } else if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return ElementKind::Edge;
    } else if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return ElementKind::Face;
    } else {
        static_assert(always_false<HandleT>::value,
                      "Unsupported PyRXMesh element kind");
    }
}

inline DType parse_dtype(const std::string& dtype)
{
    if (dtype == "float32" || dtype == "float") {
        return DType::Float32;
    }
    if (dtype == "float64" || dtype == "double") {
        return DType::Float64;
    }
    if (dtype == "int32" || dtype == "int") {
        return DType::Int32;
    }
    if (dtype == "int8") {
        return DType::Int8;
    }
    throw std::invalid_argument("Unsupported RXMesh attribute dtype: " + dtype);
}

inline const char* dtype_name(DType dtype)
{
    switch (dtype) {
        case DType::Float32:
            return "float32";
        case DType::Float64:
            return "float64";
        case DType::Int32:
            return "int32";
        case DType::Int8:
            return "int8";
        default:
            return "unknown";
    }
}

inline const char* element_kind_name(ElementKind element_kind)
{
    switch (element_kind) {
        case ElementKind::Vertex:
            return "vertex";
        case ElementKind::Edge:
            return "edge";
        case ElementKind::Face:
            return "face";
        default:
            return "unknown";
    }
}

inline rxmesh::locationT parse_location(int location)
{
    return static_cast<rxmesh::locationT>(location);
}

inline void synchronize_device_transfer(rxmesh::locationT source,
                                        rxmesh::locationT target)
{
    using namespace rxmesh;
    if (((source | target) & rxmesh::DEVICE) == rxmesh::DEVICE) {
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
    }
}

inline rxmesh::layoutT parse_layout(int layout)
{
    return static_cast<rxmesh::layoutT>(layout);
}

inline void ensure_polyscope_available()
{
#if !USE_POLYSCOPE
    throw std::runtime_error(
        "RXMesh was built with RX_USE_POLYSCOPE=OFF; visualization is not "
        "available.");
#endif
}

inline void show_polyscope()
{
    ensure_polyscope_available();
#if USE_POLYSCOPE
    polyscope::show();
#endif
}

inline void destroy_mesh_capsule(PyObject* capsule)
{
    auto* data = static_cast<pyrxmesh::MeshCapsule*>(
        PyCapsule_GetPointer(capsule, pyrxmesh::mesh_capsule_name));
    if (data) {
        delete data;
    } else {
        PyErr_Clear();
    }
}

inline void destroy_attribute_capsule(PyObject* capsule)
{
    auto* data = static_cast<pyrxmesh::AttributeCapsule*>(
        PyCapsule_GetPointer(capsule, pyrxmesh::attribute_capsule_name));
    if (data) {
        delete data;
    } else {
        PyErr_Clear();
    }
}


template <rxmesh::Op op>
struct DummyQueryLambda
{
    __device__ void operator()(typename rxmesh::InputHandle<op>::type,
                               typename rxmesh::IteratorType<op>::type) const
    {
    }
};

template <uint32_t blockThreads>
const void* dummy_query_kernel(rxmesh::Op op)
{
    switch (op) {
        case rxmesh::Op::VV:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::VV,
                                             DummyQueryLambda<rxmesh::Op::VV>>);
        case rxmesh::Op::VE:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::VE,
                                             DummyQueryLambda<rxmesh::Op::VE>>);
        case rxmesh::Op::VF:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::VF,
                                             DummyQueryLambda<rxmesh::Op::VF>>);
        case rxmesh::Op::FV:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::FV,
                                             DummyQueryLambda<rxmesh::Op::FV>>);
        case rxmesh::Op::FE:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::FE,
                                             DummyQueryLambda<rxmesh::Op::FE>>);
        case rxmesh::Op::FF:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::FF,
                                             DummyQueryLambda<rxmesh::Op::FF>>);
        case rxmesh::Op::EV:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::EV,
                                             DummyQueryLambda<rxmesh::Op::EV>>);
        case rxmesh::Op::EE:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::EE,
                                             DummyQueryLambda<rxmesh::Op::EE>>);
        case rxmesh::Op::EF:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::EF,
                                             DummyQueryLambda<rxmesh::Op::EF>>);
        case rxmesh::Op::EVDiamond:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<
                    blockThreads,
                    rxmesh::Op::EVDiamond,
                    DummyQueryLambda<rxmesh::Op::EVDiamond>>);
        default:
            throw std::invalid_argument(
                "Unsupported PyRXMesh query op for plugin launch.");
    }
}

template <uint32_t blockThreads>
void fill_plugin_launch_box(rxmesh::RXMeshStatic* mesh,
                            rxmesh::Op            op,
                            const void*,
                            bool                       oriented,
                            pyrxmesh::PluginLaunchBox* out)
{
    rxmesh::LaunchBox<blockThreads> launch_box;
    mesh->prepare_launch_box(
        {op}, launch_box, dummy_query_kernel<blockThreads>(op), oriented);

    out->blocks                   = launch_box.blocks;
    out->num_threads              = launch_box.num_threads;
    out->num_registers_per_thread = launch_box.num_registers_per_thread;
    out->smem_bytes_dyn           = launch_box.smem_bytes_dyn;
    out->smem_bytes_static        = launch_box.smem_bytes_static;
    out->local_mem_per_thread     = launch_box.local_mem_per_thread;
}

inline void prepare_plugin_launch_box(rxmesh::RXMeshStatic* mesh,
                                      rxmesh::Op            op,
                                      uint32_t              block_threads,
                                      const void*           kernel,
                                      bool                  oriented,
                                      void*                 launch_box)
{
    auto* out = static_cast<pyrxmesh::PluginLaunchBox*>(launch_box);
    switch (block_threads) {
        case 128:
            fill_plugin_launch_box<128>(mesh, op, kernel, oriented, out);
            return;
        case 256:
            fill_plugin_launch_box<256>(mesh, op, kernel, oriented, out);
            return;
        case 384:
            fill_plugin_launch_box<384>(mesh, op, kernel, oriented, out);
            return;
        case 512:
            fill_plugin_launch_box<512>(mesh, op, kernel, oriented, out);
            return;
        case 768:
            fill_plugin_launch_box<768>(mesh, op, kernel, oriented, out);
            return;
        case 1024:
            fill_plugin_launch_box<1024>(mesh, op, kernel, oriented, out);
            return;
        default:
            throw std::invalid_argument(
                "Unsupported PyRXMesh query block size. Use one of 128, 256, "
                "384, 512, 768, or 1024.");
    }
}

inline py::capsule mesh_capsule(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh)
{
    auto* capsule_data               = new pyrxmesh::MeshCapsule();
    capsule_data->abi_version        = plugin_abi_version;
    capsule_data->build_config       = build_config_tag;
    capsule_data->mesh               = mesh.get();
    capsule_data->context            = &mesh->get_context();
    capsule_data->prepare_launch_box = &prepare_plugin_launch_box;

    return py::capsule(
        capsule_data, pyrxmesh::mesh_capsule_name, destroy_mesh_capsule);
}


inline glm::fvec3 sequence_to_fvec3(const py::sequence& values,
                                    const char*         name)
{
    if (py::len(values) != 3) {
        throw std::invalid_argument(std::string(name) +
                                    " must contain exactly 3 values.");
    }
    return glm::fvec3(values[0].cast<float>(),
                      values[1].cast<float>(),
                      values[2].cast<float>());
}

inline py::array_t<float> vec3_to_numpy(const glm::vec3& value)
{
    py::array_t<float> out({static_cast<py::ssize_t>(3)});
    auto               view = out.mutable_unchecked<1>();
    view(0)                 = value[0];
    view(1)                 = value[1];
    view(2)                 = value[2];
    return out;
}

void register_module_core(py::module_& m);
void register_handles(py::module_& m);
void register_attribute(py::module_& m);
void register_dense_matrix(py::module_& m);
void register_geometry(py::module_& m);
void register_sparse_matrix(py::module_& m);
void register_solvers(py::module_& m);
void register_mesh(py::module_& m);


py::object make_sparse_matrix_from_mesh(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    rxmesh::Op                            op,
    std::string                           dtype);
}  // namespace pyrxmesh_py
