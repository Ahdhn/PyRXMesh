#pragma once

#include <Python.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/query_kernel.cuh"
#include "rxmesh/rxmesh_static.h"

#ifndef PYRXMESH_PLUGIN_ABI_VERSION
#define PYRXMESH_PLUGIN_ABI_VERSION 2
#endif

#ifndef PYRXMESH_BUILD_CONFIG
#define PYRXMESH_BUILD_CONFIG "unknown"
#endif

namespace pyrxmesh {

namespace py = pybind11;

inline constexpr const char* mesh_capsule_name      = "pyrxmesh.mesh.v1";
inline constexpr const char* attribute_capsule_name = "pyrxmesh.attribute.v1";

enum class ElementKind : uint32_t
{
    Vertex = 1,
    Edge   = 2,
    Face   = 3,
};

enum class DType : uint32_t
{
    Float32 = 1,
    Float64 = 2,
    Int8    = 3,
    UInt8   = 4,
    Int16   = 5,
    UInt16  = 6,
    Int32   = 7,
    UInt32  = 8,
    Int64   = 9,
    UInt64  = 10,

};

struct MeshCapsule
{
    uint32_t               abi_version;
    const char*            build_config;
    rxmesh::RXMeshStatic*  mesh;
    const rxmesh::Context* context;
    void (*prepare_launch_box)(rxmesh::RXMeshStatic* mesh,
                               rxmesh::Op            op,
                               uint32_t              block_threads,
                               const void*           kernel,
                               bool                  oriented,
                               void*                 launch_box);
};

struct AttributeCapsule
{
    uint32_t               abi_version;
    const char*            build_config;
    ElementKind            element_kind;
    DType                  dtype;
    uint32_t               num_attributes;
    rxmesh::AttributeBase* attribute;
};

struct PluginLaunchBox
{
    uint32_t blocks                   = 0;
    uint32_t num_threads              = 0;
    uint32_t num_registers_per_thread = 0;
    size_t   smem_bytes_dyn           = 0;
    size_t   smem_bytes_static        = 0;
    size_t   local_mem_per_thread     = 0;
};

template <typename T>
struct dtype_of;

template <>
struct dtype_of<float>
{
    static constexpr DType value = DType::Float32;
};

template <>
struct dtype_of<double>
{
    static constexpr DType value = DType::Float64;
};

template <>
struct dtype_of<int32_t>
{
    static constexpr DType value = DType::Int32;
};

template <>
struct dtype_of<int8_t>
{
    static constexpr DType value = DType::Int8;
};

template <typename HandleT>
struct element_kind_of;

template <>
struct element_kind_of<rxmesh::VertexHandle>
{
    static constexpr ElementKind value = ElementKind::Vertex;
};

template <>
struct element_kind_of<rxmesh::EdgeHandle>
{
    static constexpr ElementKind value = ElementKind::Edge;
};

template <>
struct element_kind_of<rxmesh::FaceHandle>
{
    static constexpr ElementKind value = ElementKind::Face;
};

inline void ensure_abi(const uint32_t abi_version, const char* build_config)
{
    if (abi_version != PYRXMESH_PLUGIN_ABI_VERSION) {
        throw std::runtime_error("PyRXMesh plugin ABI mismatch: plugin ABI " +
                                 std::to_string(PYRXMESH_PLUGIN_ABI_VERSION) +
                                 ", object ABI " + std::to_string(abi_version));
    }

    if (std::string(build_config) != std::string(PYRXMESH_BUILD_CONFIG)) {
        throw std::runtime_error(
            "PyRXMesh build configuration mismatch. Rebuild the plugin against "
            "the installed pyrxmesh package.");
    }
}

inline py::object capsule_from(py::handle object)
{
    if (!py::hasattr(object, "__rxmesh_capsule__")) {
        throw std::runtime_error(
            "Object does not provide a PyRXMesh capsule. Pass a pyrxmesh mesh "
            "or attribute object.");
    }
    return object.attr("__rxmesh_capsule__")();
}

inline MeshCapsule* mesh_capsule(py::handle object)
{
    py::object capsule = capsule_from(object);
    auto*      data    = static_cast<MeshCapsule*>(
        PyCapsule_GetPointer(capsule.ptr(), mesh_capsule_name));
    if (!data) {
        throw py::error_already_set();
    }
    ensure_abi(data->abi_version, data->build_config);
    return data;
}

inline AttributeCapsule* attribute_capsule(py::handle object)
{
    py::object capsule = capsule_from(object);
    auto*      data    = static_cast<AttributeCapsule*>(
        PyCapsule_GetPointer(capsule.ptr(), attribute_capsule_name));
    if (!data) {
        throw py::error_already_set();
    }
    ensure_abi(data->abi_version, data->build_config);
    return data;
}

inline rxmesh::RXMeshStatic& mesh(py::handle object)
{
    MeshCapsule* data = mesh_capsule(object);
    if (!data->mesh) {
        throw std::runtime_error("PyRXMesh mesh capsule contains a null mesh.");
    }
    return *data->mesh;
}

inline const rxmesh::Context& context(py::handle object)
{
    MeshCapsule* data = mesh_capsule(object);
    if (!data->context) {
        throw std::runtime_error(
            "PyRXMesh mesh capsule contains a null context.");
    }
    return *data->context;
}

template <rxmesh::Op op, uint32_t blockThreads, typename LambdaT>
void for_each(py::handle    mesh_object,
              const LambdaT user_lambda,
              const bool    oriented = false,
              cudaStream_t  stream   = NULL)
{
    MeshCapsule* data = mesh_capsule(mesh_object);
    if (!data->mesh || !data->context || !data->prepare_launch_box) {
        throw std::runtime_error(
            "PyRXMesh mesh capsule is missing query-launch data.");
    }

    PluginLaunchBox launch_box;
    data->prepare_launch_box(
        data->mesh,
        op,
        blockThreads,
        reinterpret_cast<const void*>(
            rxmesh::detail::query_kernel<blockThreads, op, LambdaT>),
        oriented,
        &launch_box);

    rxmesh::detail::query_kernel<blockThreads, op, LambdaT>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn,
           stream>>>(*data->context, oriented, user_lambda);
}

template <typename T, typename HandleT>
rxmesh::Attribute<T, HandleT>& attribute(py::handle object)
{
    AttributeCapsule* data = attribute_capsule(object);
    if (data->dtype != dtype_of<T>::value) {
        throw std::runtime_error("PyRXMesh attribute dtype mismatch.");
    }
    if (data->element_kind != element_kind_of<HandleT>::value) {
        throw std::runtime_error("PyRXMesh attribute element kind mismatch.");
    }
    if (!data->attribute) {
        throw std::runtime_error(
            "PyRXMesh attribute capsule contains a null attribute.");
    }
    return *static_cast<rxmesh::Attribute<T, HandleT>*>(data->attribute);
}

template <typename T>
rxmesh::VertexAttribute<T>& vertex_attribute(py::handle object)
{
    return attribute<T, rxmesh::VertexHandle>(object);
}

template <typename T>
rxmesh::EdgeAttribute<T>& edge_attribute(py::handle object)
{
    return attribute<T, rxmesh::EdgeHandle>(object);
}

template <typename T>
rxmesh::FaceAttribute<T>& face_attribute(py::handle object)
{
    return attribute<T, rxmesh::FaceHandle>(object);
}

inline std::string runtime_build_config()
{
    py::module_ pyrxmesh = py::module_::import("pyrxmesh");
    return pyrxmesh.attr("build_config_tag")().cast<std::string>();
}

inline void require_compatible_runtime()
{
    const std::string runtime = runtime_build_config();
    if (runtime != std::string(PYRXMESH_BUILD_CONFIG)) {
        throw std::runtime_error(
            "This PyRXMesh plugin was built against a different "
            "RXMesh/PyRXMesh configuration. Rebuild the plugin in the active "
            "environment.");
    }
}

inline void require_compatible_runtime(py::module_&)
{
    require_compatible_runtime();
}

}  // namespace pyrxmesh
