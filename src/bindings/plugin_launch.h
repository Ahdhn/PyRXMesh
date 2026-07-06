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

}  // namespace pyrxmesh_py
