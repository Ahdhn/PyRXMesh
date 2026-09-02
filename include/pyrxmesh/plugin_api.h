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
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"

#ifndef PYRXMESH_PLUGIN_ABI_VERSION
#error \
    "PYRXMESH_PLUGIN_ABI_VERSION must be provided by the PyRXMesh CMake target"
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
    Int32   = 7,
};

struct MeshCapsule
{
    uint32_t              abi_version;
    const char*           build_config;
    rxmesh::RXMeshStatic* mesh;
    int                   log_level;
};

struct AttributeCapsule
{
    uint32_t               abi_version;
    const char*            build_config;
    ElementKind            element_kind;
    DType                  dtype;
    rxmesh::AttributeBase* attribute;
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
    const std::string plugin_config(PYRXMESH_BUILD_CONFIG);
    const std::string object_config =
        build_config ? std::string(build_config) : std::string("<null>");
    if (abi_version != PYRXMESH_PLUGIN_ABI_VERSION) {
        throw std::runtime_error(
            "PyRXMesh plugin ABI mismatch: plugin ABI (expected) " +
            std::to_string(PYRXMESH_PLUGIN_ABI_VERSION) +
            ", runtime object ABI (actual) " + std::to_string(abi_version) +
            ". Plugin build config: '" + plugin_config +
            "'. Runtime object build config: '" + object_config +
            "'. Rebuild the plugin against the installed pyrxmesh package.");
    }
    if (object_config != plugin_config) {
        throw std::runtime_error(
            "PyRXMesh build configuration mismatch. Plugin build config "
            "(expected): '" +
            plugin_config + "'. Runtime object build config (actual): '" +
            object_config +
            "'. Rebuild the plugin against the installed pyrxmesh package.");
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

namespace detail {

inline void synchronize_plugin_logger(const int runtime_log_level)
{
    // Header-only RXMesh code in a plugin can have its own logger singleton.
    auto& plugin_logger = rxmesh::Log::get_logger();
    if (!plugin_logger) {
        plugin_logger = spdlog::default_logger();
    }
    if (plugin_logger == spdlog::default_logger()) {
        plugin_logger->set_level(
            static_cast<spdlog::level::level_enum>(runtime_log_level));
    }
}

}  // namespace detail

inline MeshCapsule mesh_capsule(py::handle object)
{
    py::object capsule = capsule_from(object);
    auto*      data    = static_cast<MeshCapsule*>(
        PyCapsule_GetPointer(capsule.ptr(), mesh_capsule_name));
    if (!data) {
        throw py::error_already_set();
    }
    ensure_abi(data->abi_version, data->build_config);
    const MeshCapsule result = *data;
    detail::synchronize_plugin_logger(result.log_level);
    return result;
}

inline AttributeCapsule attribute_capsule(py::handle object)
{
    py::object capsule = capsule_from(object);
    auto*      data    = static_cast<AttributeCapsule*>(
        PyCapsule_GetPointer(capsule.ptr(), attribute_capsule_name));
    if (!data) {
        throw py::error_already_set();
    }
    ensure_abi(data->abi_version, data->build_config);
    return *data;
}

inline rxmesh::RXMeshStatic& mesh(py::handle object)
{
    const MeshCapsule data = mesh_capsule(object);
    if (!data.mesh) {
        throw std::runtime_error("PyRXMesh mesh capsule contains a null mesh.");
    }
    return *data.mesh;
}

inline const rxmesh::Context& context(py::handle object)
{
    return mesh(object).get_context();
}

template <rxmesh::Op op, uint32_t blockThreads, typename LambdaT>
void for_each(py::handle    mesh_object,
              const LambdaT user_lambda,
              const bool    oriented = false,
              cudaStream_t  stream   = nullptr)
{
    mesh(mesh_object).for_each<op, blockThreads>(user_lambda, oriented, stream);
    using namespace rxmesh;
    CUDA_ERROR(cudaGetLastError());
}

template <typename LambdaT>
void for_each_vertex(py::handle    mesh_object,
                     const LambdaT user_lambda,
                     cudaStream_t  stream = nullptr)
{
    mesh(mesh_object).for_each_vertex(rxmesh::DEVICE, user_lambda, stream);
    using namespace rxmesh;
    CUDA_ERROR(cudaGetLastError());
}

template <typename LambdaT>
void for_each_edge(py::handle    mesh_object,
                   const LambdaT user_lambda,
                   cudaStream_t  stream = nullptr)
{
    mesh(mesh_object).for_each_edge(rxmesh::DEVICE, user_lambda, stream);
    using namespace rxmesh;
    CUDA_ERROR(cudaGetLastError());
}

template <typename LambdaT>
void for_each_face(py::handle    mesh_object,
                   const LambdaT user_lambda,
                   cudaStream_t  stream = nullptr)
{
    mesh(mesh_object).for_each_face(rxmesh::DEVICE, user_lambda, stream);
    using namespace rxmesh;
    CUDA_ERROR(cudaGetLastError());
}

template <typename T, typename HandleT>
rxmesh::Attribute<T, HandleT>& attribute(py::handle object)
{
    const AttributeCapsule data = attribute_capsule(object);
    if (data.dtype != dtype_of<T>::value) {
        throw std::runtime_error("PyRXMesh attribute dtype mismatch.");
    }
    if (data.element_kind != element_kind_of<HandleT>::value) {
        throw std::runtime_error("PyRXMesh attribute element kind mismatch.");
    }
    if (!data.attribute) {
        throw std::runtime_error(
            "PyRXMesh attribute capsule contains a null attribute.");
    }
    return *static_cast<rxmesh::Attribute<T, HandleT>*>(data.attribute);
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

inline void require_compatible_runtime(uint32_t           plugin_abi,
                                       const std::string& plugin_config)
{
    py::module_    pyrxmesh = py::module_::import("pyrxmesh");
    const uint32_t runtime_abi =
        pyrxmesh.attr("abi_version")().cast<uint32_t>();
    const std::string runtime_config =
        pyrxmesh.attr("build_config_tag")().cast<std::string>();
    if (runtime_abi != plugin_abi || runtime_config != plugin_config) {
        throw std::runtime_error(
            "This PyRXMesh plugin was built against a different "
            "RXMesh/PyRXMesh runtime. Plugin ABI (compiled/expected): " +
            std::to_string(plugin_abi) +
            "; runtime ABI (active/actual): " + std::to_string(runtime_abi) +
            ". Plugin build config (expected): '" + plugin_config +
            "'. Runtime build config (actual): '" + runtime_config +
            "'. Rebuild the plugin in the active environment.");
    }

    // Give inline RXMesh code in this plugin a harmless fallback logger.
    auto& plugin_logger = rxmesh::Log::get_logger();
    if (!plugin_logger) {
        plugin_logger = spdlog::default_logger();
    }
}

inline void require_compatible_runtime()
{
    require_compatible_runtime(PYRXMESH_PLUGIN_ABI_VERSION,
                               std::string(PYRXMESH_BUILD_CONFIG));
}

inline void require_compatible_runtime(py::module_&)
{
    require_compatible_runtime();
}

}  // namespace pyrxmesh
