"""PyTorch helpers for differentiable RXMesh scalar energies."""

from __future__ import annotations

import weakref

from . import HessianSparseMatrix, JacobianSparseMatrix
from ._rxmesh import ElementKind, ScalarEnergy


_ELEMENT_KIND = {
    "vertex": ElementKind.Vertex,
    "edge": ElementKind.Edge,
    "face": ElementKind.Face,
}
_LINEAR_TO_GLOBAL_INDICES = weakref.WeakKeyDictionary()
_TORCH_FUNCTION = None


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for differentiable PyRXMesh energies."
        ) from exc
    return torch


def _shape2(shape) -> tuple[int, int]:
    try:
        rows, cols = map(int, shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must contain exactly two integers") from exc
    if rows < 0 or cols <= 0:
        raise ValueError("shape must be (nonnegative rows, positive columns)")
    return rows, cols


def empty_soa(shape, *, dtype=None, device=None, requires_grad=False):
    """Allocate a logical (n, k) Tensor with RXMesh strides (1, n)."""

    torch = _require_torch()
    rows, cols = _shape2(shape)
    return torch.empty_strided(
        (rows, cols),
        (1, rows),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def _kind_from_name(element: str):
    try:
        return _ELEMENT_KIND[element]
    except KeyError as exc:
        raise ValueError("element must be 'vertex', 'edge', or 'face'") from exc


def _linear_to_global_index(mesh, element: str, device):
    torch = _require_torch()
    kind = _kind_from_name(element)
    indices = _LINEAR_TO_GLOBAL_INDICES.get(mesh)
    if indices is None:
        indices = {}
        _LINEAR_TO_GLOBAL_INDICES[mesh] = indices

    index = indices.get(kind)
    requested_device = torch.device(device)
    if index is None or index.device != requested_device:
        index = torch.as_tensor(
            mesh.linear_to_global(kind),
            dtype=torch.int64,
            device=requested_device,
        )
        indices[kind] = index
    return index


def to_linear_soa(mesh, values, *, element="vertex"):
    """Gather global-order rows into RXMesh linear order and SoA storage."""

    torch = _require_torch()
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise TypeError("values must be a two-dimensional torch.Tensor")
    index = _linear_to_global_index(mesh, element, values.device)
    if values.shape[0] != index.numel():
        raise ValueError("values row count does not match the mesh")
    output = empty_soa(values.shape, dtype=values.dtype, device=values.device)
    output.copy_(values.index_select(0, index))
    return output


def to_global_order(mesh, values, *, element="vertex"):
    """Scatter RXMesh linear rows into global order."""

    torch = _require_torch()
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise TypeError("values must be a two-dimensional torch.Tensor")
    index = _linear_to_global_index(mesh, element, values.device)
    if values.shape[0] != index.numel():
        raise ValueError("values row count does not match the mesh")
    output = torch.empty_like(values, memory_format=torch.contiguous_format)
    output.index_copy_(0, index, values)
    return output


def _validate_energy_input(energy, x, copy: str) -> None:
    torch = _require_torch()
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided:
        raise TypeError("ScalarEnergy.torch() expects a strided Tensor")
    if not x.is_cuda:
        raise ValueError("ScalarEnergy.torch() requires a CUDA Tensor")
    expected_shape = (energy.element_count, energy.variable_dim)
    if x.ndim != 2 or tuple(x.shape) != expected_shape:
        raise ValueError(
            f"input shape must be {expected_shape}; got {tuple(x.shape)}"
        )
    expected_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[energy.dtype]
    if x.dtype != expected_dtype:
        raise TypeError(
            f"input dtype must be {expected_dtype}; got {x.dtype}"
        )
    if copy not in ("auto", "never"):
        raise ValueError("copy must be 'auto' or 'never'")


def _is_soa(x) -> bool:
    return tuple(x.stride()) == (1, x.shape[0])


def _prepare_torch_input(x, copy: str):
    if _is_soa(x):
        return x
    if copy == "never":
        raise ValueError(
            "copy='never' requires RXMesh SoA strides "
            f"(1, {x.shape[0]}); got {tuple(x.stride())}"
        )
    output = empty_soa(x.shape, dtype=x.dtype, device=x.device)
    output.copy_(x)
    return output


def _stream_token(stream) -> int:
    raw = int(stream.cuda_stream)
    return 1 if raw == 0 else raw


def _run_torch_forward(energy, x, copy, with_gradient):
    torch = _require_torch()
    stream = torch.cuda.current_stream()
    effective_input = _prepare_torch_input(x, copy)
    gradient = (
        torch.empty(
            (energy.element_count, energy.variable_dim),
            dtype=x.dtype,
            device=x.device,
        )
        if with_gradient
        else None
    )
    term_losses = torch.empty(
        energy.term_count, dtype=x.dtype, device=x.device
    )

    # RXMesh's read is invisible to Torch's allocator. The output buffers are
    # created and consumed on this stream, so their normal Torch ownership is
    # sufficient.
    effective_input.record_stream(stream)
    energy._torch_forward(
        effective_input.data_ptr(),
        gradient.data_ptr() if gradient is not None else 0,
        term_losses.data_ptr(),
        _stream_token(stream),
    )
    return term_losses.sum(), gradient


def _torch_function():
    global _TORCH_FUNCTION
    if _TORCH_FUNCTION is not None:
        return _TORCH_FUNCTION

    torch = _require_torch()
    once_differentiable = torch.autograd.function.once_differentiable

    class _ScalarEnergyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, energy, copy):
            loss, gradient = _run_torch_forward(energy, x, copy, True)
            ctx.save_for_backward(gradient)
            ctx.energy = energy
            return loss

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            (gradient,) = ctx.saved_tensors
            return gradient * grad_output, None, None

    _TORCH_FUNCTION = _ScalarEnergyFunction
    return _TORCH_FUNCTION


def _energy_torch(self, x, *, copy="auto"):
    """Evaluate the energy as a first-order PyTorch autograd layer.

    copy='auto' borrows an RXMesh-SoA input and stages any other accepted
    layout. copy='never' requires the zero-copy SoA layout.
    """

    torch = _require_torch()
    _validate_energy_input(self, x, copy)
    if torch.is_grad_enabled() and x.requires_grad:
        return _torch_function().apply(x, self, copy)
    return _run_torch_forward(self, x, copy, False)[0]


ScalarEnergy.torch = _energy_torch


__all__ = [
    "HessianSparseMatrix",
    "JacobianSparseMatrix",
    "ScalarEnergy",
    "empty_soa",
    "to_global_order",
    "to_linear_soa",
]
