"""PyTorch interop helpers attached to native PyRXMesh classes.

This module is imported automatically by ``pyrxmesh/__init__.py`` and patches
``to_torch`` / ``from_torch_copy`` (and friends) onto the relevant native
classes. ``torch`` itself is imported lazily on first use, so importing this
module does not pull torch into the process.
"""

from __future__ import annotations

from . import (
    Attribute,
    DenseMatrix,
    EdgeAttributeFloat32,
    EdgeAttributeFloat64,
    EdgeAttributeInt32,
    EdgeAttributeInt8,
    FaceAttributeFloat32,
    FaceAttributeFloat64,
    FaceAttributeInt32,
    FaceAttributeInt8,
    Location,
    SparseMatrix,
    VertexAttributeFloat32,
    VertexAttributeFloat64,
    VertexAttributeInt32,
    VertexAttributeInt8,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for PyRXMesh torch interop helpers. "
            "Install torch or use the DLPack/NumPy APIs directly."
        ) from exc
    return torch


def _torch_dtype_name(dtype):
    torch = _require_torch()
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.int32:
        return "int32"
    raise TypeError(f"Unsupported torch dtype for PyRXMesh copy: {dtype}")


class _DenseMatrixDlpackView:
    """Wrap a ``DenseMatrix`` so ``torch.utils.dlpack.from_dlpack`` can
    forward its ``stream`` argument into ``DenseMatrix.to_dlpack``."""

    def __init__(self, matrix, location):
        self._matrix = matrix
        self._location = location
        self._is_host = location == Location.HOST
        self._is_device = location == Location.DEVICE

    def __dlpack__(self, stream=None):
        return self._matrix.to_dlpack(self._location, stream=stream)

    def __dlpack_device__(self):
        if self._is_host:
            return (1, 0)
        if self._is_device:
            torch = _require_torch()
            return (2, torch.cuda.current_device())
        raise ValueError(
            "DenseMatrix.to_torch() location must be Location.HOST or "
            "Location.DEVICE."
        )


class _SparseMatrixDlpackView:
    """Wrap a single ``SparseMatrix`` CSR component for
    ``torch.utils.dlpack.from_dlpack``."""

    def __init__(self, matrix, location, component):
        self._matrix = matrix
        self._location = location
        self._component = component
        self._is_host = location == Location.HOST
        self._is_device = location == Location.DEVICE

    def __dlpack__(self, stream=None):
        if self._component == "row_ptr":
            return self._matrix._row_ptr_dlpack(self._location, stream=stream)
        if self._component == "col_idx":
            return self._matrix._col_indices_dlpack(self._location, stream=stream)
        if self._component == "values":
            return self._matrix._values_dlpack(self._location, stream=stream)
        raise ValueError("Unsupported SparseMatrix DLPack component.")

    def __dlpack_device__(self):
        if self._is_host:
            return (1, 0)
        if self._is_device:
            torch = _require_torch()
            return (2, torch.cuda.current_device())
        raise ValueError(
            "SparseMatrix.to_torch() location must be Location.HOST or "
            "Location.DEVICE."
        )


def _dense_matrix_to_torch(self, location=Location.DEVICE):
    torch = _require_torch()
    return torch.utils.dlpack.from_dlpack(
        _DenseMatrixDlpackView(self, location)
    )


def _dense_matrix_from_torch_copy(source, order="col_major"):
    tensor = source.detach() if hasattr(source, "detach") else source
    return DenseMatrix.from_dlpack_copy(tensor, order=order)


def _dense_matrix_from_torch_view(source, order="col_major"):
    return DenseMatrix.from_dlpack_view(source, order=order)


def _sparse_matrix_to_torch(self, location=Location.DEVICE):
    torch = _require_torch()
    crow = torch.utils.dlpack.from_dlpack(
        _SparseMatrixDlpackView(self, location, "row_ptr")
    )
    col = torch.utils.dlpack.from_dlpack(
        _SparseMatrixDlpackView(self, location, "col_idx")
    )
    val = torch.utils.dlpack.from_dlpack(
        _SparseMatrixDlpackView(self, location, "values")
    )
    return torch.sparse_csr_tensor(
        crow,
        col,
        val,
        size=self.shape,
        device=val.device,
    )


def _sparse_matrix_from_torch_values_copy(
    self,
    values,
    target=Location.ALL,
    stream=None,
):
    tensor = values.detach() if hasattr(values, "detach") else values
    self.from_dlpack_values_copy(tensor, target=target, stream=stream)
    return self


def _sparse_matrix_from_torch_copy(source, dtype=None, stream=None):
    torch = _require_torch()
    if source.layout != torch.sparse_csr:
        raise TypeError(
            "SparseMatrix.from_torch_copy() expects a torch sparse CSR "
            "tensor."
        )
    value_dtype = dtype or _torch_dtype_name(source.values().dtype)
    return SparseMatrix.from_dlpack_copy(
        source.crow_indices().detach(),
        source.col_indices().detach(),
        source.values().detach(),
        tuple(source.shape),
        dtype=value_dtype,
        stream=stream,
    )


def _attribute_to_torch(self, location=Location.DEVICE):
    torch = _require_torch()
    return torch.utils.dlpack.from_dlpack(self.to_dlpack(location))


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

_attribute_types = (
    Attribute,
    VertexAttributeFloat32,
    VertexAttributeFloat64,
    VertexAttributeInt32,
    VertexAttributeInt8,
    EdgeAttributeFloat32,
    EdgeAttributeFloat64,
    EdgeAttributeInt32,
    EdgeAttributeInt8,
    FaceAttributeFloat32,
    FaceAttributeFloat64,
    FaceAttributeInt32,
    FaceAttributeInt8,
)
for _attribute_type in _attribute_types:
    _attribute_type.to_torch = _attribute_to_torch
    _attribute_type.from_torch_copy = _attribute_from_torch_copy
del _attribute_type, _attribute_types
