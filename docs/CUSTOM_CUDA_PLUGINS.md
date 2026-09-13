# Writing Custom CUDA Plugins

Use a plugin when an operation needs an RXMesh C++/CUDA device lambda but keep mesh setup and application logic in Python

## Create the plugin

With PyRXMesh, `nvcc`, and a compatible C++ compiler available:

```bash
python -m pyrxmesh.plugin my_kernels
cd my_kernels
python -m pip install -v --no-build-isolation .
```

The generated package already contains its Python module and build files. Most work happens in:

```text
src/my_kernels.cu
```

It initially computes edge lengths, so you can build and import it before replacing the example.

## Write a kernel

The generated function unwraps existing PyRXMesh attributes and captures their RXMesh wrappers in a device lambda:

```cpp
void compute_edge_lengths(py::object mesh_obj,
                          py::object coords_obj,
                          py::object out_obj)
{
    using pyrxmesh;
    auto coords = vertex_attribute<float>(coords_obj);
    auto out    = edge_attribute<float>(out_obj);

    for_each<Op::EV, 256>(
        mesh_obj,
        [coords, out] __device__(const EdgeHandle& eh,
                                 const VertexIterator& iter) mutable {
            const Eigen::Vector3f a = coords.to_eigen<3>(iter[0]);
            const Eigen::Vector3f b = coords.to_eigen<3>(iter[1]);
            out(eh) = (a - b).norm();
        });
}
```

`Op::EV` means "for each edge, query its vertices." The second explicit template argument is the CUDA block size.

Add each new function to the generated module:

```cpp
m.def("compute_edge_lengths",
      &compute_edge_lengths,
      py::arg("mesh"),
      py::arg("coords"),
      py::arg("out"));
```

Also import each new binding in `src/my_kernels/__init__.py` so it is available from the top level package:

```python
from ._my_kernels import compute_edge_lengths, update

__all__ = ["compute_edge_lengths", "update"]
```

Keep the generated `pyrxmesh::require_compatible_runtime(m)` call. It detects plugin ABI and recorded build-configuration mismatches.

## Call it from Python

```python
import pyrxmesh as rx
import my_kernels

rx.init(0)
mesh = rx.RXMeshStatic("mesh.obj")
coords = mesh.input_vertex_coordinates()
lengths = mesh.add_edge_attribute("lengths", dtype="float32", dim=1)

my_kernels.compute_edge_lengths(mesh, coords, lengths)
values = lengths.to_numpy_copy(source="device") #deep copy
```

The kernel writes directly to the device allocation of `lengths`. The last line is the explicit copy to NumPy host memory.

## Pass data and parameters

Use the helper matching an attribute's element type:

| Attribute | C++ helper                              |
| --------- | --------------------------------------- |
| Vertex    | `pyrxmesh::vertex_attribute<T>(object)` |
| Edge      | `pyrxmesh::edge_attribute<T>(object)`   |
| Face      | `pyrxmesh::face_attribute<T>(object)`   |

`T` must match the Python dtype, and a GPU lambda needs device storage. Attributes default to `location="all"` and `layout="soa"`, so their existing device allocation can normally be used without a copy.

Prefer ordinary typed arguments for settings:

```cpp
void update(py::object mesh_obj, py::object velocity_obj, float dt);

m.def("update", &update,
      py::arg("mesh"), py::arg("velocity"), py::arg("dt"));
```

This gives the natural Python call `my_kernels.update(mesh, velocity, dt)`. Convert Python values on the host, capture scalars and attribute wrappers by value, and make the lambda `mutable` when it writes an attribute.

## Choose an iteration

Use a unary wrapper when only the current element is needed:

```cpp
pyrxmesh::for_each_vertex(mesh_obj, vertex_lambda);
pyrxmesh::for_each_edge(mesh_obj, edge_lambda);
pyrxmesh::for_each_face(mesh_obj, face_lambda);
```

Use `pyrxmesh::for_each` for a neighborhood:

| Query                        | For each | Provides                                             |
| ---------------------------- | -------- | ---------------------------------------------------- |
| `Op::VV`, `Op::VE`, `Op::VF` | vertex   | adjacent vertices, edges, or faces                   |
| `Op::EV`, `Op::EE`, `Op::EF` | edge     | endpoint vertices, adjacent edges, or incident faces |
| `Op::FV`, `Op::FE`, `Op::FF` | face     | its vertices, edges, or adjacent faces               |

The first letter is the element being processed; the second is the queried element. See the [RXMesh documentation](https://ahdhn.github.io/RXMeshDocs/) for specialized and oriented queries.

An algorithm can make several wrapper calls on the same CUDA stream. Stream ordering preserves their order without combining them into a `__global__` kernel.

## Rebuild and debug

After editing the CUDA source:

```bash
python -m pip install -v --no-build-isolation --force-reinstall --no-deps .
```

CUDA launches are asynchronous. Keep the mesh and captured attributes alive until their work completes, and synchronize the relevant stream while locating a device error.

The complete tested package can be found under [examples/custom_kernel_plugin](examples/custom_kernel_plugin/). For details about the the internals of the plugins, check out [Plugin SDK Internals](PLUGIN_SDK.md).
