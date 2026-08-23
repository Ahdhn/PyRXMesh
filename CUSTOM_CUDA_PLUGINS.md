# Custom CUDA Plugins

Most PyRXMesh workflows should stay in Python. Custom CUDA plugins are for the parts of RXMesh that require C++/CUDA device lambdas.

The intended workflow is to keep orchestration and data management in Python, and put only the RXMesh lambda code in a small compiled plugin package.

## Creating a Plugin Package

```bash
python -m pyrxmesh.plugin init my_kernels
cd my_kernels
python -m pip install -v --no-build-isolation .
```

The scaffolder generates a Python "src-layout" tree:

```text
my_kernels/
  pyproject.toml
  CMakeLists.txt
  README.md
  src/
    my_kernels.cu             # CUDA source for the plugin
    my_kernels/__init__.py    # Python package source
  my_script.py                # your own scripts can live here
```

The generated plugin builds against the `pyrxmesh` package installed in the active Python environment. It reuses the RXMesh headers and library installed with PyRXMesh, so plugin builds should compile only the plugin code instead of rebuilding RXMesh. The plugin also checks the PyRXMesh/RXMesh build configuration at runtime.

External plugins must discover the installed PyRXMesh CMake package and create their extension with `pyrxmesh_add_plugin`. Build the plugin in `Release` mode with the same host compiler and CUDA toolchain used by the installed runtime.

After `pip install`, `my_kernels` is in your environment's site-packages and importable from any directory, exactly like `pyrxmesh` itself.

### Why src-layout?

Python's `sys.path[0]` is always the directory containing the script you ran. If a plugin's Python source package sits next to that script (the natural place to put it), `import my_kernels` resolves to the **source** directory rather than the installed wheel. Because the compiled CUDA extension (`_my_kernels.<ext>`) lives only in the installed wheel, the import then fails.

The src-layout sidesteps this by tucking Python sources under `src/my_kernels/`. The plugin root contains no importable `my_kernels/` directory, so `import my_kernels` always resolves to site-packages with the compiled extension alongside it. No `sys.path` munging or `cd` discipline is required.


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

The supported custom launch surface is `pyrxmesh::for_each` with an RXMesh device lambda. The wrapper unwraps the Python mesh and delegates to `RXMeshStatic::for_each`. RXMesh prepares the launch for the actual `rxmesh::detail::query_kernel<blockThreads, op, LambdaT>` specialization, so its shared-memory, register, and occupancy diagnostics describe the plugin lambda that will run. Attribute and mesh storage remain in place; the launch does not add a Python-side data copy.

Each `pyrxmesh::for_each` call represents one RXMesh query operation. Compose an algorithm that needs multiple operations as multiple lambda launches on the same CUDA stream (or omit the stream argument on every call to use the same default stream):

```cpp
pyrxmesh::for_each<Op::EV, 256>(
    mesh_obj, ev_lambda, false, stream);
pyrxmesh::for_each<Op::FV, 256>(
    mesh_obj, fv_lambda, false, stream);
```

CUDA stream ordering sequences the operations without an intervening host synchronization. Any raw `cudaStream_t` supplied to `pyrxmesh::for_each` must belong to the mesh's CUDA device. Launches are asynchronous, so the mesh and every attribute or other storage captured by a lambda must remain alive until the launch has completed on that stream.

## Examples

The repository includes a complete custom-kernel plugin example in [`examples/custom_kernel_plugin`](examples/custom_kernel_plugin) using the same src-layout the scaffolder emits:

```text
examples/custom_kernel_plugin/
  pyproject.toml
  CMakeLists.txt
  run_edge_lengths.py
  src/
    rxmesh_edge_lengths.cu
    rxmesh_edge_lengths/__init__.py
```

It computes one scalar edge-length attribute on the GPU using an RXMesh `Op::EV` device lambda, then reads the result back from Python.

Build the example plugin from the PyRXMesh repository root:

```bash
python -m pip install -v --no-build-isolation examples/custom_kernel_plugin
```

After PyRXMesh is installed, this build links against the installed PyRXMesh package and should not fetch or compile RXMesh again.

Run it as:

```bash
python examples/custom_kernel_plugin/run_edge_lengths.py --input mesh.obj
```

The example package shows the intended split where Python loads the mesh and owns the workflow while `src/rxmesh_edge_lengths.cu` contains only the performance critical RXMesh lambda.


For PyRXMesh source-build and incremental-build notes, see
[DEVELOPING.md](DEVELOPING.md).
