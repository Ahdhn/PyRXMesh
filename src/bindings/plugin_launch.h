#pragma once

#include "bindings/common.h"

namespace pyrxmesh_py {

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

inline py::capsule mesh_capsule(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh)
{
    auto* capsule_data         = new pyrxmesh::MeshCapsule();
    capsule_data->abi_version  = plugin_abi_version;
    capsule_data->build_config = build_config_tag;
    capsule_data->mesh         = mesh.get();
    const auto& runtime_logger = rxmesh::Log::get_logger();
    capsule_data->log_level    = static_cast<int>(
        runtime_logger ? runtime_logger->level() : spdlog::level::info);

    return py::capsule(
        capsule_data, pyrxmesh::mesh_capsule_name, destroy_mesh_capsule);
}

}  // namespace pyrxmesh_py
