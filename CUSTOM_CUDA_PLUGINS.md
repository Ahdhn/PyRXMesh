# Custom CUDA Plugins

Most PyRXMesh workflows should stay in Python. Custom CUDA plugins are for the parts of RXMesh that require C++/CUDA device lambdas.

The intended workflow is to keep orchestration and data management in Python, and put only the hot RXMesh lambda code in a small compiled plugin package.

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

Use `pyrxmesh::for_each` for RXMesh query operations from plugin modules. It delegates launch-box preparation to the installed PyRXMesh runtime and then launches the plugin's CUDA lambda without an extra Python data copy.

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
