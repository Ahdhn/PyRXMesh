# PyRXMesh [![Build wheels](https://github.com/Ahdhn/PyRXMesh/actions/workflows/wheels.yml/badge.svg)](https://github.com/Ahdhn/PyRXMesh/actions/workflows/wheels.yml) [![PyPI](https://img.shields.io/pypi/v/pyrxmesh.svg)](https://pypi.org/project/pyrxmesh/)

PyRXMesh brings [RXMesh](https://github.com/owensgroup/RXMesh) GPU mesh processing to Python. PyRXMesh allows using Python and PyTorch to organize an application, while RXMesh accelerates the mesh processing and automatic differentiation on the GPU.

PyRXMesh provides:

- triangle meshes built from OBJ files or NumPy arrays
- vertex, edge, and face attributes on the host and GPU
- NumPy, SciPy, and PyTorch interoperability
- dense and sparse matrices and linear solvers
- small compiled plugins for custom RXMesh CUDA operations
- differentiable scalar energies that can be interop with PyTorch autograd.

## Install

```bash
python -m pip install PyRXMesh
```

PyRXMesh needs an NVIDIA GPU and a driver compatible with the CUDA version in the wheel. PyTorch and SciPy are optional. Install SciPy normally, and choose the CUDA-enabled PyTorch build for your system from the [PyTorch installation page](https://pytorch.org/get-started/locally/).

Visualization is optional and it depends on [Polyscope](https://polyscope.run/py/):

```bash
python -m pip install "PyRXMesh[viz]"
```

For a source build, see [docs/DEVELOPING.md](docs/DEVELOPING.md).

## Create a mesh

Initialize RXMesh once, then load a mesh or build one from arrays:

```python
import pyrxmesh as rx

rx.init(0) #0: GPU id

vertices, faces = rx.create_plane(32, 32, dx=0.05)
mesh = rx.RXMeshStatic(vertices, faces)

print(mesh.num_vertices, mesh.num_edges, mesh.num_faces)

velocity = mesh.add_vertex_attribute("velocity", dtype="float32", dim=3)
velocity.reset(0, "device")
```

`RXMeshStatic("mesh.obj")` loads an OBJ file. Mesh and attribute arrays use RXMesh linear order by default and maps are available when an application needs input/global order.

See [examples/load_and_show.py](examples/load_and_show.py) for mesh inspection and Polyscope visualization, and the tests examples on [mesh construction and export](tests/test_geometry_io.py) and [attributes handling](tests/test_attributes.py).

## Automatic differentiation and PyTorch interop

A *differentiable plugin* defines one or more scalar energy terms as RXMesh CUDA lambdas. We provide a starter plugin/python module  which creates an independent project with boilerplate code (`pyproject.toml`, `CMakeLists.txt`, etc) needed to compile the CUDA lambda function and build their python bindings:

```bash
python -m pyrxmesh.diff_plugin my_energy
```

The user writes their energy terms in `.cu` file. Once built, PyRXMesh evaluates its value and gradient on the GPU and exposes the result to PyTorch:

```python
import torch
import my_energy

energy = my_energy.make_energy(mesh, {"weight": 0.5})
x = torch.as_tensor(
    mesh.vertices(), dtype=torch.float32, device="cuda"
).requires_grad_()

loss = energy.torch(x)
loss.backward()
```

The gradient flows through whatever PyTorch computation produced `x`. For an optimizer that accepts a value and gradient directly, use `energy.value_and_grad()` instead of creating an autograd node.

The starter contains the CUDA term, Python package, build files, and run script. See [Writing Differentiable Energy Plugins](DIFFERENTIABLE_ENERGY_PLUGINS.md) for the complete workflow, including query selection, fixed attributes, row order, and zero-copy input.

## Exchange data with Python

Attributes are attached to vertices, edges, or faces. They use SoA layout by default and may have host storage, device storage, or both.

```python
coords = mesh.input_vertex_coordinates()

host_view = coords.to_numpy("host") #Zero-copy numpy view
device_tensor = coords.to_torch("device") #Zero-copy pytorch view 
owned_array = coords.to_numpy_copy(source="host") #deep copy 
```

Unsuffixed `to_*` methods return a zero-copy view when supported. Methods ending in `_copy` return independent storage. Host and device allocations are separate; use `attribute.move(source, target)` when one side must receive changes from the other.

Mesh ordering maps, face arrays, and the Polyscope edge permutation are available directly from `RXMeshStatic`. See [tests/test_core_mesh.py](tests/test_core_mesh.py) and [tests/test_attributes.py](tests/test_attributes.py) for the supported operations.

## Add custom CUDA operations

When an operation cannot be expressed efficiently from Python, generate a small plugin and write only the RXMesh device lambda:

```bash
python -m pyrxmesh.plugin my_kernels
```

The generated package builds against the PyRXMesh installation in the active environment. See [Writing Custom CUDA Plugins](docs/CUSTOM_CUDA_PLUGINS.md) and [examples/custom_kernel_plugin](examples/custom_kernel_plugin/).

## Matrices and solvers

PyRXMesh exposes dense matrices, CSR sparse matrices, mesh-derived sparsity patterns, iterative solvers, and optional direct solvers. Dtypes, storage locations, matrix order, and mesh query patterns use short string arguments where they apply.

<!--Working examples are collected in the tests for [dense matrices](tests/test_dense_matrix.py), [sparse matrices](tests/test_sparse_matrix.py), and [solvers](tests/test_solvers.py).-->

## Documentation

- [Writing Differentiable Energy Plugins](docs/DIFFERENTIABLE_ENERGY_PLUGINS.md)
- [Writing Custom CUDA Plugins](docs/CUSTOM_CUDA_PLUGINS.md)
- [Developing PyRXMesh](docs/DEVELOPING.md)
- [RXMesh CUDA/C++ documentation](https://ahdhn.github.io/RXMeshDocs/)
