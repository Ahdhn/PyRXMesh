# PyRXMesh: Python binding and package for [RXMesh](https://github.com/owensgroup/RXMesh)

## **Build**

### Prerequisite

You need the platform CUDA/C++ toolchain:

- Windows: Visual Studio C++ tools plus NVIDIA CUDA Toolkit with `nvcc`.
- Linux: GCC/Clang compatible with your CUDA Toolkit, NVIDIA CUDA Toolkit with
  `nvcc`, and the OpenGL/X11 development packages required by Polyscope.

### Create conda environment 

```bash
conda env create -f environment.yml
conda activate PyRXMesh
```

### Install from a Specific RXMesh Git Tag

The RXMesh version is selected with a scikit-build config setting:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

Examples:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=main
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=v0.2.1
```

To build against a fork:

```bash
python -m pip install -v . \
  -Ccmake.define.PYRXMESH_RXMESH_GIT_REPOSITORY=https://github.com/<user>/RXMesh.git \
  -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

For local RXMesh development without fetching from Git:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_SOURCE_DIR=C:/path/to/RXMesh
```

### Run the Example

Examples are not part of the installed package, so editing them does not require reinstalling/recompiling PyRXMesh:

```bash
python examples/load_and_show.py --input mesh.obj
```

In Python, import the package as `pyrxmesh`:

```python
import pyrxmesh as rx

rx.init(0)
mesh = rx.RXMeshStatic("mesh.obj")
rx.show()
```

## Attributes From Python

PyRXMesh exposes RXMesh attributes so Python can create data that compiled CUDA plugins operate on:

```python
import pyrxmesh as rx

mesh = rx.RXMeshStatic("mesh.obj")

coords = mesh.input_vertex_coordinates()
edge_lengths = mesh.add_edge_attribute("edge_lengths", dtype="float32", dim=1)
velocity = mesh.add_vertex_attribute("velocity", dtype="float32", dim=3)

velocity.reset(0.0, rx.Location.DEVICE)
print(edge_lengths.to_numpy())  # copies from DEVICE to a NumPy array
```

Supported dtypes are `float32`, `float64`, `int32`, and `int8`.

## Custom CUDA Kernel Plugins

RXMesh computations often rely on CUDA device lambdas, which cannot be supplied directly from Python. The intended workflow is to keep orchestration and data management in Python, and put only the hot RXMesh lambda code in a small compiled plugin package.

Scaffold a plugin:

```bash
python -m pyrxmesh.plugin init my_kernels
cd my_kernels
python -m pip install -v --no-build-isolation .
```

The generated plugin builds against the `pyrxmesh` package installed in the active Python environment. It uses the same RXMesh repository/tag/options and checks the build configuration at runtime.

Generated Python usage:

```python
import pyrxmesh as rx
import my_kernels

mesh = rx.RXMeshStatic("mesh.obj")
coords = mesh.input_vertex_coordinates()
edge_lengths = mesh.add_edge_attribute("edge_lengths", dtype="float32", dim=1)

my_kernels.compute_edge_lengths(mesh, coords, edge_lengths)
```

The corresponding CUDA code receives PyRXMesh objects and unwraps them through `pyrxmesh/plugin_api.h`:

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

Use the console script form if preferred:

```bash
pyrxmesh-plugin init my_kernels
```

### Plugin Example

The repository includes a complete custom-kernel plugin example in `examples/custom_kernel_plugin`. It computes one scalar edge-length attribute on the GPU using an RXMesh `Op::EV` device lambda, then reads the result back from Python.

Build the example plugin from the PyRXMesh repository root:

```bash
python -m pip install -v --no-build-isolation examples/custom_kernel_plugin
```

Run it as:

```bash
python examples/custom_kernel_plugin/run_edge_lengths.py --input mesh.obj 
```

The example package shows the intended split, i.e., Python loads the mesh and owns the workflow, while `src/rxmesh_edge_lengths.cu` contains only the performance critical RXMesh lambda.

## Incremental Builds

Builds reuse a persistent scikit-build-core build directory:

```text
build/{wheel_tag}
```

The first install for a Python/platform/RXMesh tag combination is a full build. Subsequent installs with the same settings should be incremental. For the fastest developer loop, disable build isolation after installing the build requirements into the conda environment:

```bash
python -m pip install scikit-build-core pybind11
python -m pip install -v --no-build-isolation . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

To force a clean rebuild, delete the matching `build/<wheel_tag>` directory.