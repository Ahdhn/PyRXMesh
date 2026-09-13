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

For creating another energy, choosing terms and queries, and understanding ordering and copies, see [Writing Differentiable Energy Plugins](../../docs/DIFFERENTIABLE_ENERGY_PLUGINS.md).
