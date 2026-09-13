# Custom Kernel Plugin Example

This installable example adds one CUDA operation to Python:

```python
compute_edge_lengths(mesh, coordinates, output)
```

It evaluates each mesh edge and writes its length directly into a device edge attribute. The source is [src/rxmesh_edge_lengths.cu](src/rxmesh_edge_lengths.cu).

## Build and run

Install PyRXMesh first, then run from the repository root:

```bash
python -m pip install -v --no-build-isolation --force-reinstall --no-deps examples/custom_kernel_plugin
python examples/custom_kernel_plugin/run_edge_lengths.py --input meshes/sphere3.obj
```

The script prints the mesh size and the minimum, mean, and maximum edge length.

For creating your own package, passing attributes and parameters, and choosing an RXMesh query, see [Writing Custom CUDA Plugins](../../docs/CUSTOM_CUDA_PLUGINS.md).
