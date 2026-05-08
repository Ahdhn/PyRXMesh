# Custom CUDA Plugins

Most PyRXMesh workflows should stay in Python: load meshes, create attributes,
move data, and orchestrate processing from Python code. Custom CUDA plugins are
for the parts of RXMesh that require C++/CUDA device lambdas.

The intended workflow is to keep orchestration and data management in Python,
and put only the hot RXMesh lambda code in a small compiled plugin package.

## Scaffold a Plugin

Create a plugin package:

```bash
python -m pyrxmesh.plugin init my_kernels
cd my_kernels
python -m pip install -v --no-build-isolation .
```

The generated plugin builds against the `pyrxmesh` package installed in the
active Python environment. It uses the same RXMesh repository/tag/options and
checks the build configuration at runtime.

Use the console script form if preferred:

```bash
pyrxmesh-plugin init my_kernels
```

## Use a Plugin From Python

Generated Python usage:

```python
import pyrxmesh as rx
import my_kernels

mesh = rx.RXMeshStatic("mesh.obj")
coords = mesh.input_vertex_coordinates()
edge_lengths = mesh.add_edge_attribute("edge_lengths", dtype="float32", dim=1)

my_kernels.compute_edge_lengths(mesh, coords, edge_lengths)
```

The CUDA code receives PyRXMesh objects and unwraps them through
`pyrxmesh/plugin_api.h`:

```cpp
auto coords = pyrxmesh::vertex_attribute<float>(coords_obj);
auto out = pyrxmesh::edge_attribute<float>(out_obj);

pyrxmesh::for_each<Op::EV, 256>(
    mesh_obj,
    [coords, out] __device__(const EdgeHandle& eh,
                             const VertexIterator& iter) mutable {
        const Eigen::Vector3f a = coords.to_eigen<3>(iter[0]);
        const Eigen::Vector3f b = coords.to_eigen<3>(iter[1]);
        out(eh) = (a - b).norm();
    });
```

Use `pyrxmesh::for_each` for RXMesh query operations from plugin modules. It
delegates launch-box preparation to the installed PyRXMesh runtime and then
launches the plugin's CUDA lambda without an extra Python data copy.

## Checked-In Example

The repository includes a complete custom-kernel plugin example in
[`examples/custom_kernel_plugin`](examples/custom_kernel_plugin). It computes
one scalar edge-length attribute on the GPU using an RXMesh `Op::EV` device
lambda, then reads the result back from Python.

Build the example plugin from the PyRXMesh repository root:

```bash
python -m pip install -v --no-build-isolation examples/custom_kernel_plugin
```

Run it as:

```bash
python examples/custom_kernel_plugin/run_edge_lengths.py --input mesh.obj
```

The example package shows the intended split: Python loads the mesh and owns the
workflow, while `src/rxmesh_edge_lengths.cu` contains only the performance
critical RXMesh lambda.
