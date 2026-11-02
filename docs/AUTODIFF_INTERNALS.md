# Autodiff Internals

This document describes the implementation and ownership rules behind PyRXMesh scalar energies. Energy authors normally need only [Writing Differentiable Energy Plugins](DIFFERENTIABLE_ENERGY_PLUGINS.md).

## Architecture

A compiled plugin constructs

```cpp
pyrxmesh::diff::ScalarGradientProblem<T, VariableDim, HandleT, WithHessian, ProjectHessian>
```

and registers RXMesh scalar terms. Both boolean template parameters default to false. The object derives from the type-erased `ScalarEnergyBase`, which is exposed to Python as `ScalarEnergy`. Hessian-enabled specializations instantiate second-order active scalars and native VV Hessian storage; the passive and gradient-only kernels are also available.

The layers are:

```text
energy.torch() / value_and_grad() / evaluate()
                    |
             ScalarEnergy binding
                    |
      ScalarGradientProblem<T, k, Handle>
                    |
       RXMesh DiffScalarProblem and terms
```

Templates and device lambdas remain in the external CUDA module. The installed PyRXMesh runtime owns the Python API, input adaptation, output buffers, and stream/lifetime coordination.

## Problem memory policy

`scalar_gradient_memory_options()` constructs the upstream problem with a PyRXMesh-specific policy:

- the optimization variable has registered metadata but no owned storage;
- its expected layout is SoA;
- the reusable internal gradient starts at `LOCATION_NONE`;
- term-loss working storage is on the device; and
- internal attribute names are unique.

The metadata-only optimization variable is important. During a Torch call, its device pointer is temporarily attached to the selected Torch input and detached before control returns. PyRXMesh therefore does not allocate or copy a second `n * VariableDim` optimization-variable buffer.

The reusable internal gradient is also omitted at construction. It is needed only by the low-level `evaluate(Attribute)` API and is allocated lazily on that API's first call.

## Torch forward path

The Python implementation is in [pyrxmesh/diff.py](../../pyrxmesh/diff.py), and the native bridge is in [include/pyrxmesh/diff_plugin_api.h](../../include/pyrxmesh/diff_plugin_api.h).

For every call, Python:

1. validates the tensor's shape, dtype, CUDA placement, and copy policy
2. borrows an exact SoA input or stages it into SoA storage
3. selects PyTorch's current CUDA stream
4. allocates or receives the output gradient
5. allocates one Torch term-loss vector
6. calls the private native `_torch_forward` entry point with raw pointers and the stream token

An input is borrowed when its logical shape is `(n, k)` and its strides are `(1, n)`. With `copy="auto"`, another strided layout is copied once on the GPU into `empty_soa` storage. With `copy="never"`, that layout mismatch is an error.

The native call attaches the selected input pointer to the metadata-only RXMesh attribute. Gradient evaluation constructs a non-owning row-major `DenseMatrix` view over the Torch output and passes that view to `eval_terms_grad_only`. RXMesh writes directly into Torch-owned memory. Term reductions are written into a Torch-owned vector, which Python sums to produce the scalar loss. Output allocation is not a data copy.

After enqueuing the work, the native layer detaches the input pointer and records the energy's last-use event. Detaching changes only metadata; it does not wait for the kernel or invalidate the underlying Torch allocation.

## `energy.torch()`

When gradients are enabled and `x.requires_grad` is true, a custom `torch.autograd.Function` requests both loss and gradient. Each forward receives a distinct gradient Tensor and saves it for backward. This is required because two outstanding graphs, `retain_graph=True`, or a later forward must not overwrite an earlier graph's derivative.

Backward multiplies the saved `dE/dx` by the incoming scalar derivative. It does not call RXMesh again. The function is marked once-differentiable, so double backward is intentionally unsupported.

When autograd is disabled or the input does not require gradients, `_torch_forward` receives a null gradient pointer and RXMesh runs the passive loss path. No gradient Tensor or internal gradient buffer is allocated.

PyTorch may reuse released allocations through its caching allocator, but PyRXMesh does not maintain a private output pool for autograd. Correct graph lifetime takes priority over reusing a particular Tensor object.

## `value_and_grad()`

`value_and_grad()` uses the same native forward but bypasses autograd. The caller supplies a row-major contiguous CUDA `out` Tensor, and RXMesh writes the gradient into it directly. Reusing that Tensor across optimizer evaluations avoids creating a fresh gradient object on every call.

The output must not alias the input and must not require gradients. Every call overwrites it. If `gradient_mask` is present, Python multiplies the completed gradient by the mask on the same stream; RXMesh still evaluates the full energy and derivative.

The loss remains a newly produced scalar Torch Tensor because it is formed from the current term-loss reduction.

## Stream and lifetime rules

Torch calls run on `torch.cuda.current_stream()`. Python calls `record_stream` for the effective input and any gradient output, including caller-owned `value_and_grad(out=...)` storage, so the Torch allocator cannot reuse those allocations before the queued RXMesh work finishes.

A `ScalarGradientProblem` records a CUDA event after each Torch forward. The next use of that same energy waits for the event on its chosen stream. This permits sequential use of one energy from different streams without a host synchronization. It does not make concurrent native calls from different host threads safe.

The autograd context retains both the saved gradient and the energy. The energy retains its mesh. Its destructor synchronizes and destroys any outstanding last-use event. PyRXMesh assumes one active GPU and does not install a CUDA device guard around every call.

Device execution is asynchronous, so faults can still surface at a later Torch or CUDA synchronization.

## Ordering helpers

RXMesh computation uses linear element order. `to_linear_soa` gathers global rows with `mesh.linear_to_global(...)` into SoA storage. Because the gather and copy are Torch operations, autograd maps a later gradient back to the original input automatically.

`to_global_order` performs the inverse scatter for manually handled results. This is necessary for a `value_and_grad` caller whose surrounding application uses global rows, because that API does not create an autograd graph to perform the inverse mapping.

## Low-level Attribute path

`energy.evaluate(attribute, derivatives=1)` exists for Attribute-based workflows. It:

1. verifies that the Attribute belongs to the same mesh and matches the energy contract;
2. lazily allocates the reusable device gradient when `derivatives >= 1`;
3. runs passive terms for order 0, gradient-only terms for order 1, or full active terms for order 2; and
4. reduces the loss to a host scalar, synchronizing the call.

`gradient_view` wraps the internal gradient without copying it and retains the energy as its external owner. The view is read-only through the PyRXMesh matrix wrapper, but its contents are overwritten by the next successful low-level evaluation. A Tensor already exported from the view cannot be made generation-aware.

`gradient_snapshot()` allocates an owned row-major device matrix, copies the current internal gradient into it, and synchronizes the selected stream before returning. It is the stable low-level result at the cost of a device-to-device copy.

The Torch path never saves this reusable internal gradient in an autograd graph.

## Assembled Hessians

`WithHessian=true` enables the second-order path for vertex variables and fixed `V`/`EV`/`FV` stencils. `ProjectHessian` is forwarded to RXMesh's `add_term` template. Projection changes local Hessians before assembly, leaving values and gradients unchanged.

After an Attribute evaluation, `has_evaluated`, `has_gradient`, and `has_hessian` record which results are available. Value-only evaluation invalidates both derivative accessors; first-order evaluation invalidates Hessian access. Torch evaluations continue using their own output buffers and do not publish new low-level result state.

`hessian_view` wraps the internal `HessianSparseMatrix<T,k>` without allocating or copying CSR buffers. Its aliasing shared pointer retains the energy, and existing device DLPack/Torch exports share the row pointers, column indices, and values. The view is mutable: edits affect the energy's Hessian, and subsequent second-order evaluations overwrite its values. Attribute evaluation is already synchronous, so acquiring the view adds no synchronization. Access requires an available second-order result.

The sparse wrapper marks this matrix as an external view, so `release()` and wrapper destruction do not free the energy's storage. Exported tensors retain the wrapper and therefore the energy even after the original Python references are deleted. Already exported views remain aliases; obtaining a new view after a lower-order evaluation is rejected. This API targets device use and does not maintain host coherence after later device writes.

`hessian_snapshot()` remains the independent-copy option. It constructs a `PyHessianSparseMatrix<T,k>` with the identical fixed VV pattern and copies device values to both its host and device allocations, synchronizing the selected stream. `evaluate_hessian(attribute)` combines second-order evaluation with that snapshot. RXMesh's `copy_from` copies values only, so this reconstruction is restricted to fixed stencils without expanded/contact sparsity.

Hessian DOFs use `k * linear_vertex_id + component`, independent of input SoA storage. Native assembly handles query-local indices and mesh handles. Global/input ordering can be applied to both matrix axes by the caller in Python; no new permutation implementation is required in PyRXMesh.

## Validation

Python validates public tensor shape, dtype, device placement, output layout, and mask shape. The native Attribute path validates mesh identity, element kind, dimension, dtype, and device storage. Capsule and runtime compatibility are covered by the [Plugin SDK](PLUGIN_SDK.md).

These tests can be found under 

- [tests/test_diff_autograd.py](../../tests/test_diff_autograd.py) checks an independent analytic reference, copies, outstanding graphs, streams, ordering, masks, and output reuse
- [examples/diff_energy_plugin](../../examples/diff_energy_plugin/) is the external build test
- [benchmarks/benchmark_diff_autograd.py](../../benchmarks/benchmark_diff_autograd.py) compares the supported evaluation paths
