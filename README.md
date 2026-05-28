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
  -Ccmake.define.PYRXMESH_RXMESH_GIT_REPO=https://github.com/<user>/RXMesh.git \
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
print(edge_lengths.to_numpy_copy(source=rx.Location.DEVICE))
```

Supported dtypes are `float32`, `float64`, `int32`, and `int8`.

## Advanced: Custom CUDA Plugins

Most PyRXMesh workflows should stay in Python. For RXMesh operations that require CUDA device lambdas, PyRXMesh also supports small compiled plugin packages that operate on PyRXMesh meshes and attributes without extra Python-side copies. Plugins build against the installed PyRXMesh package and reuse its RXMesh build, so plugin rebuilds are much smaller after PyRXMesh is installed.

See [CUSTOM_CUDA_PLUGINS.md](CUSTOM_CUDA_PLUGINS.md) for the plugin workflow.