# Developing PyRXMesh

This guide is for building PyRXMesh from source, testing changes, and working against RXMesh branches, forks, or local checkouts. If you only want to use PyRXMesh from Python, start with [README.md](README.md).

## Prerequisites

You need the platform CUDA/C++ toolchain:

- Windows: Visual Studio C++ tools plus NVIDIA CUDA Toolkit with `nvcc`.
- Linux: GCC/Clang compatible with your CUDA Toolkit, NVIDIA CUDA Toolkit with
  `nvcc`, and the OpenGL/X11 development packages required by Polyscope.

## Conda Environment

```bash
conda env create -f environment.yml
conda activate PyRXMesh
```

Install build requirements into the active environment when you want faster developer rebuilds without build isolation:

```bash
python -m pip install scikit-build-core pybind11 numpy
```

## Build From Source

The normal developer build uses the local default CUDA architecture behavior and the RXMesh ref configured in `pyproject.toml`:

```bash
python -m pip install -v .
```

For a faster inner loop after installing build requirements:

```bash
python -m pip install -v --no-build-isolation .
```

## Select An RXMesh Source

For release packages, the RXMesh ref should be source-controlled in `pyproject.toml`:

```toml
[tool.scikit-build.cmake.define]
PYRXMESH_RXMESH_GIT_TAG = "main"
```

For local experiments, override the RXMesh tag or commit from the command line:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

Examples:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=main
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=v0.2.1
```

To build against an RXMesh fork:

```bash
python -m pip install -v . \
  -Ccmake.define.PYRXMESH_RXMESH_GIT_REPO=https://github.com/<user>/RXMesh.git \
  -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

For local RXMesh development without fetching from Git:

```bash
python -m pip install -v  --no-build-isolation . -Ccmake.define.PYRXMESH_RXMESH_SOURCE_DIR=C:/path/to/RXMesh
```

## Release-Like Local Build

Release wheels build with Polyscope enabled and a broad CUDA architecture set:

```bash
python -m pip install -v . \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -Ccmake.define.PYRXMESH_USE_POLYSCOPE=ON
```

## Incremental Builds

Builds reuse a persistent scikit-build-core build directory:

```text
build/{wheel_tag}
```

The first install for a Python/platform/RXMesh configuration is a full build. Subsequent installs with the same settings should be incremental.

To force a clean rebuild, delete the matching `build/<wheel_tag>` directory.

## Run Examples

Examples are not part of the installed package, so editing them does not require reinstalling/recompiling PyRXMesh:

```bash
python examples/load_and_show.py --input mesh.obj
```

## Custom CUDA Plugins

Most PyRXMesh workflows should stay in Python. For RXMesh operations that require CUDA device lambdas, PyRXMesh supports small compiled plugin packages that operate on PyRXMesh meshes and attributes without extra Python-side copies.

See [CUSTOM_CUDA_PLUGINS.md](CUSTOM_CUDA_PLUGINS.md) for the plugin workflow.

## Build And Verify The Differentiable Energy Plugin

Install PyRXMesh first, then build the example as a separate package :

```bash
python -m pip install -v --no-build-isolation --no-deps examples/diff_energy_plugin
python -c "import pyrxmesh as rx, rxmesh_diff_energy; print(rx.abi_version()); print(rx.build_config_tag()); print(rxmesh_diff_energy.make_energy)"
python -m pytest -v tests/test_diff_energy_plugin.py
```

The import command is GPU-independent, i.e., it proves that the wheel contains enough headers, CMake metadata, and RXMesh libraries to compile and load an external CUDA plugin, and that the runtime ABI/configuration check passes. Creating a mesh/energy and running the numerical tests requires an NVIDIA GPU and compatible driver.

Rebuild every external plugin after changing a template-visible RXMesh header, `include/pyrxmesh/diff_plugin_api.h`, the plugin ABI, or a build option embedded in the compatibility tag. Inspect the active runtime with:

```bash
python -c "import pyrxmesh as rx; print(rx.abi_version()); print(rx.build_config_tag())"
```

The tag includes the PyRXMesh ABI/version, RXMesh commit, a hash of ABI-relevant RXMesh headers, CUDA architectures, and relevant feature options. A stale plugin must fail during import with its expected and actual configuration; do not bypass that error. Reinstall the plugin against the currently imported PyRXMesh instead.

## Differentiable Energy API

```python
loss = energy.torch(x, order="linear", copy="auto")
loss.backward()
```

Currently, we accept a CUDA Tensor of shape `(element_count, variable_dim)` whose dtype and CUDA device match the compiled energy. Tensors without gradients and calls under `torch.no_grad()` use the same asynchronous device forward path. When the input requires gradients and gradient mode is active, the returned 0-D CUDA Tensor participates in autograd. It has the same scalar dtype. Backward applies its incoming scalar `grad_output`, so scaled losses and sums of energies obey the chain rule. Only the optimization variable is differentiated, i.e., values captured by plugin lambdas and entries in the factory parameter dictionary are constants. Double backward, CPU inputs,
autocast dtype substitution, topology changes, CUDA graph capture, and `torch.compile` are not supported.

### Linear And Global Row Order

Existing numeric PyRXMesh APIs use RXMesh linear order, i.e., `mesh.vertices()`, Attribute rows for SoA/AoS/AoSoA layouts, `mesh.faces()` rows, face vertex IDs, and differentiable energy gradient rows. Order is never inferred from shape or strides.

For a handle at linear row `l`, let `linear_to_global[l]` be its `map_to_global` ID. The inverse and the forward/backward transformations are:

```C++
global_to_linear[linear_to_global[l]] = l
x_linear[l] = x_global[linear_to_global[l]]
g_global[linear_to_global[l]] = g_linear[l]
```

Thus `order="global"` gathers before forward and scatters before returning the input gradient. Vertex and face global IDs correspond to original input/file order. Edge global IDs are RXMesh's stable global edge enumeration; an OBJ input has no edge-row order to preserve. Validate ordering changes on a multi-patch mesh with a proven non-identity permutation, not only on a tiny single-patch mesh.

### Layout And Copy Policy

A conventional contiguous Torch Tensor with logical shape `(n, k)` has strides `(k, 1)`. RXMesh's borrowed input representation is SoA and requires strides
`(1, n)`. Allocate that representation with:

```python
x = rx.diff.empty_soa(
    (mesh.num_vertices, 3),
    dtype=torch.float32,
    device=f"cuda:{energy.cuda_device_id}",
    requires_grad=True,
)
assert x.stride() == (1, mesh.num_vertices)
```

`rx.diff.to_linear_soa(mesh, x_global, element="vertex")` performs the explicit global-to-linear layout conversion, and `rx.diff.to_global_order(...)` converts back at an API boundary.

| `copy` value | Required/selected behavior                                                                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"auto"`     | Borrow only when all strict conditions hold; otherwise stage on the same CUDA device.                                                                             |
| `"never"`    | Require `order="linear"`, exact strides `(1, n)`, correct shape/dtype/device, dense non-overlapping storage, and zero storage offset. Report the first violation. |
| `"always"`   | Always stage into private linear-SoA storage for isolation and repeatable profiling.                                                                              |

Global order necessarily adds a permutation, and an incompatible layout necessarily adds layout staging. Those kernels perform required mathematical or representation work and should not be described as avoidable interop copies.

### Low-Level Result Lifetime

The native debugging interface remains available:

```python
host_loss = energy.evaluate(attribute, stream=None)
ephemeral = energy.gradient_view
stable = energy.gradient_snapshot(stream=None)
```

`evaluate` accepts native SoA, AoS, and AoSoA Attributes that match the energy's mesh, element kind, dimension, dtype, and device. It returns a Python float and therefore is a synchronous compatibility/debugging path. `gradient_view` aliases the mutable internal row-major gradient and is valid only until the next successful evaluation. PyRXMesh makes native mutation through that view unavailable, but it cannot intercept reads from a Torch Tensor after the view has been exported through DLPack. Never save it as autograd state. `gradient_snapshot()` returns an owned row-major `DenseMatrix`
that remains stable after later evaluations. Loss, view, and snapshot access before a successful evaluation raise `RuntimeError`; use `low_level_evaluation_id` when diagnosing which result is current. The legacy `.gradient` name is only a deprecated alias for `.gradient_view`.

### Streams, Ownership, And Trusted Plugins

`energy.torch` schedules RXMesh on Torch's current CUDA stream. Each forward owns its output gradient and loss storage, so a later forward cannot overwrite an earlier graph's saved state. The energy retains its original mesh and CUDA device; changing the process's current device later does not redirect its allocations or kernels.

One native energy serializes its calls across streams with CUDA events because its terms and scratch state are shared. Different energies may overlap only where shared-mesh query/context use is proven safe. The asynchronous owner set includes input/output Tensors, the energy, mesh, terms, reducers, scratch allocations, DLPack capsules, and completion events. Normal and partial-enqueue exception paths keep those owners alive through completion before detaching or running a deleter.

Argument validation and immediate CUDA launch failures are reported by the `energy.torch(...)` call. CUDA execution remains asynchronous, so a fault that occurs after a successful launch generally appears at a later CUDA/PyTorch synchronization point. The fast path intentionally does not add a
host synchronization solely to convert such faults into immediate Python exceptions.

A compiled plugin is trusted native code. Energy lambdas must treat the optimization-variable Attribute as read-only. Writing through a borrowed pointer bypasses Torch's version counter and is undefined behavior. Use `copy="always"` when isolation is more important than avoiding input staging;
it is not a sandbox for otherwise unsafe native code.