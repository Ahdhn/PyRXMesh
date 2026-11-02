# Differentiable Energy Plugin Example

This installable example exposes an RXMesh scalar energy to PyTorch:

```text
w = w0 + w1
E(x) = w * sum_(u,v in edges) ||x[u] - x[v]||^2
```

The input is a three-component `float32` vertex field. `make_energy` registers separate `w0` and `w1` terms. The CUDA source is [src/rxmesh_diff_energy.cu](src/rxmesh_diff_energy.cu).

## Build and run

Install PyRXMesh first, then run from the repository root:

```bash
python -m pip install -v --no-build-isolation --force-reinstall --no-deps examples/diff_energy_plugin
python examples/diff_energy_plugin/run_diff_energy.py --input meshes/sphere3.obj
```

The script evaluates the energy with `energy.torch(x)` and prints the loss and gradient shape.

To enable exact or projected Hessians:

```python
energy = rxmesh_diff_energy.make_energy(
    mesh, {"w0": 0.5, "w1": 0.25},
    with_hessian=True, projected_hessian=False,
)
# positions is a 3-component float32 vertex Attribute on this mesh.
loss = energy.evaluate(positions, derivatives=0)  # Value only.
loss = energy.evaluate(positions, derivatives=1)  # Value and gradient (default).
loss = energy.evaluate(positions, derivatives=2)  # Value, gradient, Hessian.
H = energy.hessian_view.to_torch("device")  # Mutable, zero-copy device CSR.
# An independent copy is also available:
H_copy = energy.hessian_snapshot()
# Or compute and return an owned Hessian in one call:
H_rx = energy.evaluate_hessian(positions)
```

`projected_hessian=True` selects RXMesh's C++ projection before local terms are assembled. All three evaluation modes remain available and projection leaves the loss and gradient unchanged. 

`hessian_view` retains the energy and shares its device arrays. Changes through the view modify the internal Hessian; later second-order evaluations overwrite the same values. Use a snapshot when the result must stay independent.

The GPU integration test checks evaluation modes, analytic derivatives, finite differences, ordering, projection, and ownership:

```bash
python -m pytest tests/test_diff_hessian.py -q
```

For creating another energy, choosing terms and queries, and understanding ordering and copies, see [Writing Differentiable Energy Plugins](../../docs/DIFFERENTIABLE_ENERGY_PLUGINS.md).
