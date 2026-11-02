# Writing Differentiable Energy Plugins

A differentiable plugin defines a scalar energy with RXMesh CUDA lambdas. PyRXMesh evaluates values and derivatives on the GPU. PyTorch can propagate the first derivative through the computation that produced the input, and the Attribute API can return an assembled sparse Hessian.

The API is CUDA-only and supports scalar energies. Hessian computation and positive-definite projection are opt-in; Torch autograd remains first-order.

## Create and run a plugin

```bash
python -m pyrxmesh.diff_plugin my_energy
cd my_energy
python -m pip install -v --no-build-isolation .
python run_energy.py path/to/mesh.obj
```

The generated package is already complete. Start by editing `add_terms` in `src/my_energy.cu`, then reinstall it:

```bash
python -m pip install -v --no-build-isolation --force-reinstall --no-deps .
```

Rebuild the plugin after rebuilding PyRXMesh or whenever import reports a compatibility mismatch.

## Use the energy from PyTorch

```python
import torch
import pyrxmesh as rx
import my_energy

rx.init(0)
mesh = rx.RXMeshStatic("model.obj")
energy = my_energy.make_energy(mesh, {"weight": 0.5})

x = torch.as_tensor(
    mesh.vertices(), dtype=torch.float32, device="cuda"
).requires_grad_()

loss = energy.torch(x)
loss.backward()
```

`loss` is a scalar CUDA Tensor, and `x.grad` has the same shape as `x`. Evaluation uses PyTorch's current CUDA stream. Calling `.item()` or moving the result to the CPU synchronizes.

When autograd is disabled or `x.requires_grad` is false, `energy.torch(x)` evaluates only the loss.

## Define the optimization variable

The generated source defines the energy's input and Hessian capabilities at compile time:

```cpp
using Problem =
    pyrxmesh::diff::ScalarGradientProblem<
        float, 3, VertexHandle, WithHessian, ProjectHessian>;
```

This means `torch.float32`, three components, and one row per vertex. The factory selects the compiled boolean specialization using `with_hessian` and `projected_hessian`. Both default to false. A first-order edge- or face-based problem can use `EdgeHandle` or `FaceHandle`; assembled Hessians currently require vertex variables.

If these inputs change, update all of the following together:

- `ScalarGradientProblem<T, VariableDim, HandleT>`;
- `make_scalar_energy<T, VariableDim, HandleT>`;
- every `active<VariableDim>` read and fixed-size Eigen type; and
- the Python Tensor's dtype, row count, and component count.

The generated example runner initializes three-component vertex positions. Replace that initialization for another element kind or dimension.

## Write an energy term

The example starter evaluates a quadratic energy once per edge:

```cpp
template <typename Problem>
void add_terms(Problem& problem, py::dict params)
{
    const float weight =
        params.contains("weight") ? params["weight"].cast<float>() : 1.0f;

    problem.template add_term<Op::EV>(
        [weight] __device__(const auto& eh,
                            const auto& iter,
                            auto&       x) {
            using ActiveT = ACTIVE_TYPE(eh);
            const Eigen::Vector3<ActiveT> x0 =
                x.template active<3>(eh, iter, 0);
            const Eigen::Vector3<ActiveT> x1 =
                x.template active<3>(eh, iter, 1);
            return weight * (x0 - x1).squaredNorm();
        });
}
```

It contributes `weight * ||x[u] - x[v]||^2` for every owned edge. For another term:

- return one scalar from a CUDA `__device__` lambda;
- read optimized values through `x.active<VariableDim>(...)`;
- keep variable-dependent values in `ActiveT`;
- treat `x` as read-only because it may alias a Torch Tensor;
- capture only device-safe scalars and RXMesh attribute wrappers; and
- protect the domains of operations such as division, normalization, square root, determinant, and logarithm.

Multiple `add_term` calls are additive. RXMesh accumulates one loss and one complete derivative for the energy.

Hessian-enabled plugins currently accept fixed vertex stencils `V`, `EV`, and `FV`, with variable dimension 1, 2, 3, 4, or 6. Unsupported stencils fail at compile time. Gradient-only plugins retain the existing query support.

## Choose an RXMesh query

The first letter names the element that owns the term; a second letter names its neighborhood:

| Query            | Runs once per      | Provides                                        |
| ---------------- | ------------------ | ----------------------------------------------- |
| `V`, `E`, `F`    | vertex, edge, face | the owner only                                  |
| `VV`, `VE`, `VF` | vertex             | adjacent vertices or incident elements          |
| `EV`, `EE`, `EF` | edge               | endpoints, neighboring edges, or incident faces |
| `FV`, `FE`, `FF` | face               | its vertices, edges, or adjacent faces          |

A unary term has no iterator:

```cpp
problem.add_term<Op::V>(
    [weight] __device__(const auto& vh, auto& x) {
        return weight * x.active<3>(vh).squaredNorm();
    });
```

## Understand order and copies

Energy inputs use RXMesh linear row order. `mesh.vertices()` and PyRXMesh Attribute arrays already use it.

The default `copy="auto"` accepts an ordinary row-major CUDA Tensor. It borrows exact RXMesh SoA storage and otherwise stages the input on each evaluation. When either Torch API computes a derivative, RXMesh writes it directly into Torch-owned output storage.

For data from another mesh loader, gather global rows before evaluation:

```python
x_linear = rx.diff.to_linear_soa(mesh, x_global, element="vertex")
energy.torch(x_linear, copy="never").backward()
# Autograd propagates through the gather to whatever produced x_global.
```

For repeated evaluation, allocate SoA storage once and forbid staging:

```python
x = rx.diff.empty_soa(
    (mesh.num_vertices, 3), dtype=torch.float32, device="cuda"
)
x.copy_(torch.as_tensor(mesh.vertices(), device="cuda"))
x.requires_grad_()

loss = energy.torch(x, copy="never")
```

For shape `(n, k)`, SoA strides are `(1, n)`; a nonzero storage offset is allowed. `copy="never"` rejects an incompatible shape, dtype, CPU placement, or stride. Under the single-GPU policy, the caller is responsible for using the CUDA device on which RXMesh was initialized.

Vertex and face global rows correspond to input order. Edge global IDs are RXMesh's stable enumeration because OBJ files do not define an edge-row order.

## Choose an evaluation API

| API                               | Use it when                                          |
| --------------------------------- | ---------------------------------------------------- |
| `energy.torch(x)`                 | The energy is part of a PyTorch objective.           |
| `energy.value_and_grad(x, out=g)` | A numerical optimizer accepts a reusable derivative. |
| `energy.evaluate(attribute)`      | Debugging through the low-level Attribute API.       |

A direct optimizer can reuse one output:

```python
gradient = torch.empty_like(x, memory_format=torch.contiguous_format)
loss = energy.value_and_grad(x, out=gradient, copy="never")
x.grad = gradient
```

`value_and_grad()` bypasses autograd, so do not call `loss.backward()`. Every call overwrites `out`. It must be a row-major contiguous Tensor with the energy's shape and dtype, on the same CUDA device as `x`, and it must not require gradients. The caller must also ensure that `out` does not alias `x`.

The output is in linear order. If a manual optimizer stores global-order variables, convert it with `rx.diff.to_global_order(...)`. An optional `gradient_mask` of shape `(n,)` or `(n, 1)` also uses linear order and must be on `x.device`. It masks the returned derivative after evaluation; it does not change the loss, skip derivative work, or implement optimizer constraints by itself.

`energy.evaluate(attribute)` returns a synchronized Python `float` and defaults to computing its gradient. Its reusable gradient-view lifetime is documented in [Autodiff Internals](AUTODIFF_INTERNALS.md).

## Choose the derivative order and retrieve a Hessian

The same Hessian-enabled energy supports all three evaluation modes:

```python
energy = my_energy.make_energy(mesh, {"weight": 0.5}, with_hessian=True)
positions = mesh.add_vertex_attribute(
    "positions", dtype="float32", dim=3, location="all", layout="soa"
)
positions.from_numpy_copy(mesh.vertices(), target="all")

loss = energy.evaluate(positions, derivatives=0)  # Energy only.
loss = energy.evaluate(positions, derivatives=1)  # Energy and gradient.
gradient = energy.gradient_snapshot().to_torch("device")
loss = energy.evaluate(positions, derivatives=2)  # Energy, gradient, Hessian.
H_rx = energy.hessian_view
H = H_rx.to_torch("device")  # Zero-copy Torch sparse CSR view.
```

`derivatives=0` runs passive kernels and does not allocate or compute a gradient. `derivatives=1` runs gradient-only kernels even on a Hessian-enabled energy. `derivatives=2` requires `with_hessian=True`. To evaluate and retrieve an owned Hessian in one call, use `H_rx = energy.evaluate_hessian(positions)`; the loss and gradient are then available through `energy.loss` and `gradient_snapshot()`.

`has_gradient` and `has_hessian` describe the most recent successful Attribute evaluation. A lower-order evaluation invalidates higher-order accessors, preventing retrieval of stale derivatives. `hessian_view` borrows the existing device CSR storage without copying and retains the energy. Both the wrapper and its exported tensors are mutable: edits affect the energy's matrix, and subsequent second-order evaluations update the same values. Existing views remain aliases, including after the Python energy reference is deleted.

Use `H_rx = energy.hessian_snapshot()` when an independent result is needed. Snapshots remain unchanged across later evaluations and energy destruction. This method copies device values into owned host and device storage and synchronizes its stream. The zero-copy view targets GPU use; it does not automatically refresh host values after subsequent device writes.

The existing Torch APIs retain their behavior, including on Hessian-enabled energies: `energy.torch(x)` evaluates only values under `torch.no_grad()` or when `x.requires_grad` is false, and computes first derivatives when required. `value_and_grad` computes first derivatives. Neither API computes a Hessian or enables Torch double backward.

### Projected Hessians

```python
energy = my_energy.make_energy(
    mesh, {"weight": 0.5}, with_hessian=True, projected_hessian=True
)
energy.evaluate(positions, derivatives=2)
H = energy.hessian_view.to_torch("device")
```

`projected_hessian` selects RXMesh's C++ `ProjectHess` template argument. RXMesh projects each local term Hessian to positive definite before assembly, using its eigenvalue floor (1e-6 for float32, 1e-9 for float64). The loss and gradient are unchanged; the resulting matrix is an optimization approximation rather than the exact second derivative. Setting `projected_hessian=True` without `with_hessian=True` is an error. These options are fixed at construction, and exposed as `hessian_enabled` and `projected_hessian` properties.

### Hessian ordering

All Hessians use RXMesh's native linear vertex order, with scalar degree of freedom `k * linear_vertex_id + component` and shape `(n*k, n*k)`. This matches logical row-major flattening of the `(n, k)` gradient, regardless of the input Attribute/Tensor's storage layout. No internal global-order or solver permutation is applied.

Applications using input/global vertex order can convert both axes in Python. For a SciPy CSR matrix:

```python
import numpy as np

g2l = np.asarray(mesh.global_to_linear(rx.ElementKind.Vertex), dtype=np.int64)
k = energy.variable_dim
q = (k * g2l[:, None] + np.arange(k)).reshape(-1)
H_linear = H_rx.to_scipy_csr()
H_global = H_linear[q][:, q]
```

This keeps the matrix sparse. `rx.diff.to_global_order` handles `(n, k)` fields and does not convert Hessians. Solver fill-reducing permutations are separate from this mesh/input mapping.

Run `python -m pytest tests/test_diff_hessian.py -q` for the GPU integration test covering all evaluation modes, an analytic Hessian, finite differences, caller-side permutation, projection, and snapshot ownership.

The complete tested package is [examples/diff_energy_plugin](examples/diff_energy_plugin/). For details about buffer ownership, streams, and the native implementation, check out [Autodiff Internals](AUTODIFF_INTERNALS.md).
