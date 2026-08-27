"""Lazy PyTorch helpers attached to PyRXMesh's matrix types."""

from __future__ import annotations

from . import Attribute, DenseMatrix, Location, SparseMatrix


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for PyRXMesh Torch interop."
        ) from exc
    return torch


def _torch_dtype_name(dtype):
    torch = _require_torch()
    names = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int32: "int32",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported torch dtype: {dtype}") from exc


def _device(owner, location):
    if location == Location.HOST:
        return (1, 0)
    if location == Location.DEVICE:
        return owner.__dlpack_device__()
    raise ValueError("location must be Location.HOST or Location.DEVICE")


class _DlpackView:
    def __init__(self, exporter, device):
        self._exporter = exporter
        self._device = device

    def __dlpack__(self, stream=None):
        return self._exporter(stream)

    def __dlpack_device__(self):
        return self._device


def _view(owner, location, exporter):
    return _DlpackView(exporter, _device(owner, location))


def _dense_matrix_to_torch(self, location=Location.DEVICE):
    torch = _require_torch()
    return torch.utils.dlpack.from_dlpack(
        _view(
            self,
            location,
            lambda stream: self.to_dlpack(location, stream=stream),
        )
    )


def _dense_matrix_from_torch_copy(source, order="col_major"):
    tensor = source.detach() if hasattr(source, "detach") else source
    return DenseMatrix.from_dlpack_copy(tensor, order=order)


def _dense_matrix_from_torch_view(source, order="col_major"):
    return DenseMatrix.from_dlpack_view(source, order=order)


def _sparse_matrix_to_torch(self, location=Location.DEVICE):
    torch = _require_torch()
    device = (
        (1, 0)
        if location == Location.HOST
        else (2, torch.cuda.current_device())
        if location == Location.DEVICE
        else None
    )
    if device is None:
        raise ValueError("location must be Location.HOST or Location.DEVICE")

    def convert(exporter):
        return torch.utils.dlpack.from_dlpack(
            _DlpackView(
                lambda stream: exporter(location, stream=stream), device
            )
        )

    values = convert(self._values_dlpack)
    return torch.sparse_csr_tensor(
        convert(self._row_ptr_dlpack),
        convert(self._col_indices_dlpack),
        values,
        size=self.shape,
        device=values.device,
    )


def _sparse_matrix_from_torch_values_copy(
    self, values, target=Location.ALL, stream=None
):
    tensor = values.detach() if hasattr(values, "detach") else values
    self.from_dlpack_values_copy(tensor, target=target, stream=stream)
    return self


def _sparse_matrix_from_torch_copy(source, dtype=None, stream=None):
    torch = _require_torch()
    if source.layout != torch.sparse_csr:
        raise TypeError("source must be a torch sparse CSR tensor")
    return SparseMatrix.from_dlpack_copy(
        source.crow_indices().detach(),
        source.col_indices().detach(),
        source.values().detach(),
        tuple(source.shape),
        dtype=dtype or _torch_dtype_name(source.values().dtype),
        stream=stream,
    )


def _attribute_to_torch(self, location=Location.DEVICE):
    torch = _require_torch()
    return torch.utils.dlpack.from_dlpack(
        _view(
            self,
            location,
            lambda stream: self.to_dlpack(location, stream=stream),
        )
    )


def _attribute_from_torch_copy(self, values, target=Location.ALL):
    tensor = values.detach() if hasattr(values, "detach") else values
    self.from_dlpack_copy(tensor, target=target)
    return self


DenseMatrix.to_torch = _dense_matrix_to_torch
DenseMatrix.from_torch_copy = staticmethod(_dense_matrix_from_torch_copy)
DenseMatrix.from_torch_view = staticmethod(_dense_matrix_from_torch_view)
SparseMatrix.to_torch = _sparse_matrix_to_torch
SparseMatrix.from_torch_copy = staticmethod(_sparse_matrix_from_torch_copy)
SparseMatrix.from_torch_values_copy = _sparse_matrix_from_torch_values_copy
Attribute.to_torch = _attribute_to_torch
Attribute.from_torch_copy = _attribute_from_torch_copy
