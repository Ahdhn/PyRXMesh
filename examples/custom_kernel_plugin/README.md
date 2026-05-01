# Custom Kernel Plugin Example

This example is a small installable Python package that adds one CUDA/RXMesh
kernel to Python: `compute_edge_lengths(mesh, coords, out)`.

The Python script handles orchestration, mesh loading, attribute creation, and
result readback. The C++/CUDA plugin contains only the RXMesh device lambda that
needs to run without Python overhead.

## Build

From the PyRXMesh repository root, after installing PyRXMesh into the active
conda environment:

```bash
python -m pip install -v --no-build-isolation examples/custom_kernel_plugin
```

The plugin uses `python -m pyrxmesh.cmake_dir` to find the CMake package
installed with PyRXMesh, so it is built against the same RXMesh source/tag and
build options as the active environment.

## Run

```bash
python examples/custom_kernel_plugin/run_edge_lengths.py \
  --input C:/path/to/RXMesh/input/cube.obj \
  --device-id 0
```

Expected output is a short mesh summary plus min/mean/max edge lengths.

## Kernel

The core of the plugin is:

```cpp
pyrxmesh::for_each<Op::EV, 256>(
    mesh_obj,
    [coords, out] __device__(const EdgeHandle& eh,
                             const VertexIterator& iter) mutable {
        const Eigen::Vector3f a = coords.to_eigen<3>(iter[0]);
        const Eigen::Vector3f b = coords.to_eigen<3>(iter[1]);
        out(eh) = (a - b).norm();
    });
```

`pyrxmesh::for_each` is the plugin-safe wrapper around RXMesh query launches.
It keeps launch setup compatible with the installed PyRXMesh runtime while the
custom CUDA lambda still lives in the plugin.
