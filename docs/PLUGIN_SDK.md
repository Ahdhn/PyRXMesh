# Plugin SDK Internals

This document describes how external CUDA plugins are built and connected to a running PyRXMesh installation. It is for PyRXMesh maintainers and people debugging the plugin. For the normal authoring workflow, see [Writing Custom CUDA Plugins](CUSTOM_CUDA_PLUGINS.md).

## Installed SDK

PyRXMesh wheels include the pieces needed to compile a plugin:

- PyRXMesh and RXMesh headers;
- the RXMesh library and its CMake package;
- `pyrxmeshConfig.cmake`; and
- the `pyrxmesh::plugin_api` interface target.

Generated plugins use the active Python environment for the package directory:

```cmake
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pyrxmesh.cmake_dir
    OUTPUT_VARIABLE pyrxmesh_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(pyrxmesh CONFIG REQUIRED PATHS "${pyrxmesh_DIR}" NO_DEFAULT_PATH) pyrxmesh_add_plugin(_my_plugin src/my_plugin.cu)
```

`pyrxmesh_add_plugin` links the installed `RXMesh::RXMesh` target, adds the installed headers and compatibility definitions, enables CUDA separable compilation and device linking, and applies the CUDA options RXMesh lambdas need. It also inherits the CUDA architectures recorded by the PyRXMesh build unless the plugin selects them explicitly.

A plugin should never find or compile a second RXMesh checkout independently. Its headers, device code, and linked library must describe the same build.

Python mesh and attribute objects expose small named capsules defined in [include/pyrxmesh/plugin_api.h](../../include/pyrxmesh/plugin_api.h). A capsule contains:

- the plugin ABI version
- the PyRXMesh build configuration tag
- a raw RXMesh object pointer
- attribute dtype and element kind metadata

The helper functions `pyrxmesh::mesh`, `vertex_attribute<T>`, `edge_attribute<T>`, and `face_attribute<T>` validate the capsule before returning the native object. They do not copy data and do not transfer ownership. Python owners must remain alive until asynchronous work using those objects has finished.

Every module must call this while it initializes:

```cpp
PYBIND11_MODULE(_my_plugin, m)
{
    pyrxmesh::require_compatible_runtime(m);
    // bindings...
}
```

This rejects an incompatible module before a mesh or attribute pointer crosses the boundary.

## ABI and build compatibility

`PYRXMESH_PLUGIN_ABI_VERSION` is defined once in the root `CMakeLists.txt` and is propagated through the installed CMake target. Change it only when an existing compiled plugin can no longer use the capsule or C++ plugin API safely. Ordinary compatible additions do not require an ABI bump.

The build tag records the ABI, PyRXMesh version, configured RXMesh repository/ref string, precision and optional-solver settings, and CUDA architectures. A plugin records the same tag when it is compiled. Import fails if either the ABI or tag differs from the active runtime.

## Query launch path

The neighborhood wrapper uses RXMesh directly:

```cpp
pyrxmesh::for_each<Op::EV, 256>(mesh_obj, lambda, oriented, stream);
```

It calls `RXMeshStatic::for_each<Op::EV, 256>` with the user's actual lambda type. RXMesh therefore prepares its `LaunchBox` for the real `query_kernel` specialization and reports the relevant register, shared-memory, occupancy, and launch statistics. Do not substitute a generic query kernel when preparing this launch.

The unary wrappers use RXMesh's device traversal directly:

```cpp
pyrxmesh::for_each_vertex(mesh_obj, vertex_lambda, stream);
pyrxmesh::for_each_edge(mesh_obj, edge_lambda, stream);
pyrxmesh::for_each_face(mesh_obj, face_lambda, stream);
```

All wrappers run on the GPU. They use RXMesh's `CUDA_ERROR` helper to report an immediate launch error after dispatch. CUDA execution remains asynchronous, so an execution fault can still appear at a later synchronization.

`blockThreads` is part of the compiled kernel and can change occupancy and register pressure. Start with 256, as the scaffold does, until measurement or RXMesh resource diagnostics justify another value.

## Streams and multiple operations

The default stream argument is `nullptr`. A plugin with a native `cudaStream_t` can pass it as the last argument; neighborhood traversal places `oriented` immediately before it.

PyRXMesh deliberately exposes one lambda launch per wrapper call. An algorithm with several operations should call the wrappers several times on the same stream. CUDA stream ordering preserves the sequence without requiring the user to write a combined `__global__` kernel.

Launches are asynchronous. A binding that accepts or creates non-default streams is responsible for keeping every captured mesh and attribute alive and for arranging synchronization with Python frameworks. The simple public plugin workflow uses the default stream.

## Maintaining the scaffolds

The shared package templates live in `pyrxmesh/_plugin_scaffold.py`. Both entry points add native and Python package sources:

- `pyrxmesh/plugin.py` for ordinary kernels;
- `pyrxmesh/diff_plugin.py` for scalar energies.

When the SDK API changes:

1. update the shared template where possible
2. update the matching repository example
3. regenerate a temporary plugin and build it against an installed PyRXMesh
4. rebuild both example plugins
5. run their Python smoke tests
6. decide whether the change is ABI-compatible before touching the ABI version

## Common failures

**Build configuration mismatch:** PyRXMesh was rebuilt but the plugin was not. Reinstall the plugin without reusing its old build directory.

**Undefined CUDA device symbols:** the plugin was linked against the wrong RXMesh package, or its CUDA architecture does not overlap the installed library.

**Implausible LaunchBox resource values:** confirm the launch is prepared through `RXMeshStatic::for_each` with the actual lambda specialization.

**An error appears after the Python call returns:** the launch was asynchronous. Synchronize the relevant stream while debugging to locate the failing operation.
