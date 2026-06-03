# PyRXMesh

Python bindings for [RXMesh](https://github.com/owensgroup/RXMesh), a CUDA/C++ library for GPU mesh processing.

PyRXMesh allows you to keep mesh workflows in Python while using RXMesh data structures and GPU kernels underneath. It supports exchanging arrays and attributes with NumPy and PyTorch, building sparse matrices, and solving linear systems. We use a Python plugin to allow CUDA kernels to be written, compiled, and used directly in Python. Check out [CUSTOM_CUDA_PLUGINS.md](CUSTOM_CUDA_PLUGINS.md) for more details.


## Installation

Install PyRXMesh with:

```bash
python -m pip install PyRXMesh
```

PyRXMesh is a CUDA package. You need an NVIDIA driver compatible with the CUDA version used by the wheel. For source builds and development setup, see [DEVELOPING.md](DEVELOPING.md).

## Simple Example

```python
import pyrxmesh as rx

#Use GPU 0 
rx.init(0) 

mesh = rx.RXMeshStatic("mesh.obj")
print(mesh.num_vertices, mesh.num_edges, mesh.num_faces)

rx.show()
```

The repository includes a small viewer example:

```bash
python examples/load_and_show.py --input mesh.obj
```

## Mesh Attributes

Attributes are data attached to vertices, edges, or faces

```python
import pyrxmesh as rx

mesh = rx.RXMeshStatic("mesh.obj")

coords = mesh.input_vertex_coordinates()
velocity = mesh.add_vertex_attribute("velocity", dtype="float32",
                                     dim=3, location=rx.Location.ALL)

velocity.reset(0.0, rx.Location.DEVICE)

host_coords = coords.to_numpy_copy(source=rx.Location.HOST)
print(host_coords.shape)
```

## NumPy And Torch Interop

Unsuffixed `to_*` methods return zero-copy views when possible. Methods ending
in `_copy` make an explicit copy.

```python
import pyrxmesh as rx

mesh = rx.RXMeshStatic("mesh.obj")
attr = mesh.add_vertex_attribute("temperature", dtype="float32",
                                 dim=1, location=rx.Location.ALL)

attr.reset(1.0, rx.Location.ALL)

view = attr.to_numpy(rx.Location.HOST)
view[0, 0] = 42.0

owned = attr.to_numpy_copy(source=rx.Location.HOST)
print(owned[0, 0])
```

Torch views use DLPack:

```python
tensor = attr.to_torch(rx.Location.DEVICE)
tensor += 2.0
```

## Dense And Sparse Matrices

PyRXMesh exposes RXMesh dense matrices and CSR sparse matrices.

```python
import numpy as np
import pyrxmesh as rx

matrix = rx.DenseMatrix(4, 3, dtype="float32", location=rx.Location.ALL)
values = np.arange(12, dtype=np.float32).reshape(4, 3)

matrix.from_numpy_copy(values, target=rx.Location.ALL)
print(matrix.norm2())
```

Sparse matrices are CSR-only:

```python
mesh = rx.RXMeshStatic("mesh.obj")
laplace_like = mesh.sparse_matrix(rx.Op.VV, dtype="float32")

row_ptr, col_idx, values = laplace_like.to_numpy_copy()
print(laplace_like.shape, laplace_like.nnz)
```

You can multiply sparse matrices by dense vectors:

```python
x = np.ones((laplace_like.cols, 1), dtype=np.float32)
y = laplace_like.multiply_vector(x)
print(y.to_numpy_copy(source=rx.Location.HOST))
```

## Solvers

RXMesh solver bindings work with `SparseMatrix` and `DenseMatrix` objects:

```python
import numpy as np
import pyrxmesh as rx

row_ptr = np.array([0, 2, 5, 7], dtype=np.int32)
col_idx = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
values = np.array([4, 1, 1, 3, 1, 1, 2], dtype=np.float32)

A = rx.SparseMatrix.from_numpy_copy(row_ptr, col_idx, values,
                                    shape=(3, 3), dtype="float32")

b = rx.DenseMatrix(3, 1, dtype="float32", location=rx.Location.ALL)
b.from_numpy_copy(np.array([[1], [2], [3]], dtype=np.float32))

solver = rx.CGSolver(A, unknown_dim=1)
x = solver.solve(b)

print(x.to_numpy_copy(source=rx.Location.HOST))
```

## Documentation
- [RXMeshDocs](https://ahdhn.github.io/RXMeshDocs/): RXMesh CUDA/C++ documentation website 
- [DEVELOPING.md](DEVELOPING.md): build PyRXMesh from source and work against RXMesh forks, tags, or local checkouts.
- [CUSTOM_CUDA_PLUGINS.md](CUSTOM_CUDA_PLUGINS.md): write small compiled CUDA plugins that operate on PyRXMesh meshes and attributes.
