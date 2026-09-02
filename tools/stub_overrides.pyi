"""Typing contracts that cannot be recovered from pybind11 docstrings.

Classes in this file are partial: generation replaces only the members named
here and keeps every other member emitted by ``pybind11-stubgen``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import torch
from scipy.sparse import csr_matrix

from ._typing import (
    BlockShape as _BlockShape,
    CanonicalDType as _CanonicalDType,
    CanonicalDenseOrder as _CanonicalDenseOrder,
    CanonicalFloatDType as _CanonicalFloatDType,
    CanonicalNumericDType as _CanonicalNumericDType,
    CopyPolicy as _CopyPolicy,
    DenseOrder as _DenseOrder,
    DTypeName as _DTypeName,
    ElementName as _ElementName,
    FloatDType as _FloatDType,
    IntLike as _IntLike,
    LayoutArg as _LayoutArg,
    LayoutName as _LayoutName,
    LocationArg as _LocationArg,
    LocationName as _LocationName,
    MeshOrder as _MeshOrder,
    NumericDType as _NumericDType,
    OpArg as _OpArg,
    OpResultName as _OpResultName,
    Permutation as _Permutation,
    Scalar as _Scalar,
    Shape2D as _Shape2D,
    Stream as _Stream,
)

_Array: TypeAlias = npt.NDArray[Any]
_FloatArray: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]
_IndexArray: TypeAlias = npt.NDArray[np.uint32]
_HandleArray: TypeAlias = npt.NDArray[np.uint64]
_SparseArrays: TypeAlias = tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    _Array,
]
_BasicHandle: TypeAlias = VertexHandle | EdgeHandle | FaceHandle


class _LocationNamespace:
    NONE: Literal[0]
    HOST: Literal[1]
    DEVICE: Literal[2]
    ALL: Literal[15]


class _LayoutNamespace:
    AoS: Literal[0]
    AoSoA: Literal[1]
    SoA: Literal[2]


Location: Final[_LocationNamespace]
Layout: Final[_LayoutNamespace]


class Attribute:
    @property
    def dtype(self) -> _CanonicalDType: ...
    @property
    def element_kind(self) -> _ElementName: ...
    @property
    def shape(self) -> _Shape2D: ...
    @property
    def allocated(self) -> _LocationName: ...
    @property
    def layout(self) -> _LayoutName: ...

    def reset(self, value: _Scalar, location: _LocationArg = "all") -> None: ...
    def move(self, source: _LocationArg, target: _LocationArg) -> None: ...
    def copy_from(
        self,
        other: Attribute,
        source: _LocationArg = "all",
        target: _LocationArg = "all",
    ) -> None: ...
    def to_numpy(self, location: _LocationArg = "host") -> _Array: ...
    def to_numpy_copy(self, source: _LocationArg = "host") -> _Array: ...
    def from_numpy_copy(
        self,
        values: npt.ArrayLike,
        target: _LocationArg = "all",
    ) -> None: ...
    def to_torch(self, location: _LocationArg = "device") -> torch.Tensor: ...
    def from_torch_copy(
        self,
        values: torch.Tensor,
        target: _LocationArg = "all",
    ) -> Attribute: ...
    def to_dlpack(
        self,
        location: _LocationArg = "device",
        stream: _Stream = None,
    ) -> object: ...
    def from_dlpack_copy(
        self,
        source: object,
        target: _LocationArg = "all",
    ) -> None: ...
    def __dlpack__(self, stream: _Stream = None) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def to_matrix_copy(self) -> DenseMatrix: ...
    def from_matrix_copy(
        self,
        matrix: DenseMatrix,
        target: _LocationArg = "all",
    ) -> None: ...
    def argmax(
        self,
        column: _IntLike | None = None,
    ) -> tuple[_BasicHandle, _Scalar]: ...
    def argmin(
        self,
        column: _IntLike | None = None,
    ) -> tuple[_BasicHandle, _Scalar]: ...


class DenseMatrix:
    @overload
    def __init__(
        self,
        rows: _IntLike,
        cols: _IntLike,
        dtype: _NumericDType = "float32",
        location: _LocationArg = "all",
        order: _DenseOrder = "col_major",
    ) -> None: ...
    @overload
    def __init__(
        self,
        mesh: RXMeshStatic,
        rows: _IntLike,
        cols: _IntLike,
        dtype: _NumericDType = "float32",
        location: _LocationArg = "all",
        order: _DenseOrder = "col_major",
    ) -> None: ...

    @staticmethod
    def from_dlpack_copy(
        source: object,
        order: _DenseOrder = "col_major",
    ) -> DenseMatrix: ...
    @staticmethod
    def from_dlpack_view(
        source: object,
        order: _DenseOrder = "col_major",
    ) -> DenseMatrix: ...
    @staticmethod
    def from_torch_copy(
        source: torch.Tensor,
        order: _DenseOrder = "col_major",
    ) -> DenseMatrix: ...
    @staticmethod
    def from_torch_view(
        source: torch.Tensor,
        order: _DenseOrder = "col_major",
    ) -> DenseMatrix: ...

    @property
    def shape(self) -> _Shape2D: ...
    @property
    def dtype(self) -> _CanonicalNumericDType: ...
    @property
    def order(self) -> _CanonicalDenseOrder: ...
    @property
    def location(self) -> _LocationName: ...

    def move(
        self,
        source: _LocationArg,
        target: _LocationArg,
        stream: _Stream = None,
    ) -> None: ...
    def release(self, location: _LocationArg = "all") -> None: ...
    def reset(
        self,
        value: _Scalar,
        location: _LocationArg = "all",
        stream: _Stream = None,
    ) -> None: ...
    def to_numpy(self, location: _LocationArg = "host") -> _Array: ...
    def to_numpy_copy(self, source: _LocationArg = "host") -> _Array: ...
    def from_numpy_copy(
        self,
        values: npt.ArrayLike,
        target: _LocationArg = "all",
        stream: _Stream = None,
    ) -> None: ...
    def to_torch(self, location: _LocationArg = "device") -> torch.Tensor: ...
    def copy_from(
        self,
        other: DenseMatrix,
        source: _LocationArg = "all",
        target: _LocationArg = "all",
        stream: _Stream = None,
    ) -> None: ...
    def to_dlpack(
        self,
        location: _LocationArg = "device",
        stream: _Stream = None,
    ) -> object: ...
    def __dlpack__(self, stream: _Stream = None) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...


class SparseMatrix:
    def __init__(
        self,
        mesh: RXMeshStatic,
        op: _OpArg = "vv",
        dtype: _NumericDType = "float32",
    ) -> None: ...

    @staticmethod
    def from_numpy_copy(
        row_ptr: npt.ArrayLike,
        col_idx: npt.ArrayLike,
        values: npt.ArrayLike,
        shape: _Shape2D,
        dtype: _NumericDType = "float32",
    ) -> SparseMatrix: ...
    @staticmethod
    def from_dlpack_copy(
        row_ptr: object,
        col_idx: object,
        values: object,
        shape: _Shape2D,
        dtype: _NumericDType | Literal[""] = "",
        stream: _Stream = None,
    ) -> SparseMatrix: ...
    @staticmethod
    def from_torch_copy(
        source: torch.Tensor,
        dtype: _NumericDType | None = None,
        stream: _Stream = None,
    ) -> SparseMatrix: ...

    @property
    def shape(self) -> _Shape2D: ...
    @property
    def dtype(self) -> _CanonicalNumericDType: ...
    @property
    def index_dtype(self) -> Literal["int32"]: ...
    @property
    def op(self) -> _OpResultName: ...
    @property
    def location(self) -> _LocationName: ...

    def move(
        self,
        source: _LocationArg,
        target: _LocationArg,
        stream: _Stream = None,
    ) -> None: ...
    def reset(
        self,
        value: _Scalar,
        location: _LocationArg = "all",
        stream: _Stream = None,
    ) -> None: ...
    def copy_from(
        self,
        other: SparseMatrix,
        source: _LocationArg = "all",
        target: _LocationArg = "all",
        stream: _Stream = None,
    ) -> None: ...
    def to_numpy(self, location: _LocationArg = "host") -> _SparseArrays: ...
    def to_numpy_copy(
        self,
        source: _LocationArg = "host",
        stream: _Stream = None,
    ) -> _SparseArrays: ...
    def from_numpy_values_copy(
        self,
        values: npt.ArrayLike,
        target: _LocationArg = "all",
        stream: _Stream = None,
    ) -> None: ...
    def from_dlpack_values_copy(
        self,
        values: object,
        target: _LocationArg = "all",
        stream: _Stream = None,
    ) -> None: ...
    def from_torch_values_copy(
        self,
        values: torch.Tensor,
        target: _LocationArg = "all",
        stream: _Stream = None,
    ) -> SparseMatrix: ...
    def multiply_vector(
        self,
        vector: DenseMatrix | npt.ArrayLike,
        stream: _Stream = None,
    ) -> DenseMatrix: ...
    def to_torch(self, location: _LocationArg = "device") -> torch.Tensor: ...
    def to_scipy_csr(self) -> csr_matrix: ...
    def to_scipy_csr_copy(
        self,
        source: _LocationArg = "host",
        stream: _Stream = None,
    ) -> csr_matrix: ...
    def to_dlpack(
        self,
        location: _LocationArg = "device",
        stream: _Stream = None,
    ) -> tuple[object, object, object]: ...


class _JacobianSparseMatrixFloat32:
    @property
    def num_terms(self) -> int: ...
    def term_num_rows(self, term: _IntLike) -> int: ...
    def term_rows_range(self, term: _IntLike) -> tuple[int, int]: ...


class _JacobianSparseMatrixFloat64:
    @property
    def num_terms(self) -> int: ...
    def term_num_rows(self, term: _IntLike) -> int: ...
    def term_rows_range(self, term: _IntLike) -> tuple[int, int]: ...


class _HessianSparseMatrixFloat32Dim1:
    @property
    def variable_dim(self) -> Literal[1]: ...


class _HessianSparseMatrixFloat32Dim2:
    @property
    def variable_dim(self) -> Literal[2]: ...


class _HessianSparseMatrixFloat32Dim3:
    @property
    def variable_dim(self) -> Literal[3]: ...


class _HessianSparseMatrixFloat32Dim4:
    @property
    def variable_dim(self) -> Literal[4]: ...


class _HessianSparseMatrixFloat32Dim6:
    @property
    def variable_dim(self) -> Literal[6]: ...


class _HessianSparseMatrixFloat64Dim1:
    @property
    def variable_dim(self) -> Literal[1]: ...


class _HessianSparseMatrixFloat64Dim2:
    @property
    def variable_dim(self) -> Literal[2]: ...


class _HessianSparseMatrixFloat64Dim3:
    @property
    def variable_dim(self) -> Literal[3]: ...


class _HessianSparseMatrixFloat64Dim4:
    @property
    def variable_dim(self) -> Literal[4]: ...


class _HessianSparseMatrixFloat64Dim6:
    @property
    def variable_dim(self) -> Literal[6]: ...


_JacobianResult: TypeAlias = (
    _JacobianSparseMatrixFloat32 | _JacobianSparseMatrixFloat64
)
_HessianResult: TypeAlias = (
    _HessianSparseMatrixFloat32Dim1
    | _HessianSparseMatrixFloat32Dim2
    | _HessianSparseMatrixFloat32Dim3
    | _HessianSparseMatrixFloat32Dim4
    | _HessianSparseMatrixFloat32Dim6
    | _HessianSparseMatrixFloat64Dim1
    | _HessianSparseMatrixFloat64Dim2
    | _HessianSparseMatrixFloat64Dim3
    | _HessianSparseMatrixFloat64Dim4
    | _HessianSparseMatrixFloat64Dim6
)


def JacobianSparseMatrix(
    mesh: RXMeshStatic,
    ops: Sequence[_OpArg],
    block_shapes: Sequence[_BlockShape],
    dtype: _FloatDType = "float32",
) -> _JacobianResult: ...


def HessianSparseMatrix(
    mesh: RXMeshStatic,
    variable_dim: _IntLike = 3,
    extra_nnz_entries: _IntLike = 0,
    op: _OpArg = "vv",
    dtype: _FloatDType = "float32",
) -> _HessianResult: ...


class RXMeshStatic:
    def input_vertex_coordinates(
        self,
    ) -> VertexAttributeFloat32 | VertexAttributeFloat64: ...
    def vertices(self, order: _MeshOrder = "linear") -> _FloatArray: ...
    def faces(
        self,
        order: _MeshOrder = "linear",
    ) -> npt.NDArray[np.uint32]: ...
    def bounding_box(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
    def vertex_handles(self) -> _HandleArray: ...
    def edge_handles(self) -> _HandleArray: ...
    def face_handles(self) -> _HandleArray: ...
    def for_each_vertex(
        self,
        callback: Callable[[VertexHandle], object],
    ) -> None: ...
    def for_each_edge(
        self,
        callback: Callable[[EdgeHandle], object],
    ) -> None: ...
    def for_each_face(
        self,
        callback: Callable[[FaceHandle], object],
    ) -> None: ...
    def linear_to_global(self, element_kind: ElementKind) -> _IndexArray: ...
    def global_to_linear(self, element_kind: ElementKind) -> _IndexArray: ...

    @overload
    def add_vertex_attribute(
        self,
        name: str,
        dtype: Literal["float32", "float"] = "float32",
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> VertexAttributeFloat32: ...
    @overload
    def add_vertex_attribute(
        self,
        name: str,
        dtype: Literal["float64", "double"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> VertexAttributeFloat64: ...
    @overload
    def add_vertex_attribute(
        self,
        name: str,
        dtype: Literal["int32", "int"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> VertexAttributeInt32: ...
    @overload
    def add_vertex_attribute(
        self,
        name: str,
        dtype: Literal["int8"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> VertexAttributeInt8: ...
    @overload
    def add_vertex_attribute(
        self,
        name: str,
        dtype: _DTypeName,
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> Attribute: ...

    @overload
    def add_edge_attribute(
        self,
        name: str,
        dtype: Literal["float32", "float"] = "float32",
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> EdgeAttributeFloat32: ...
    @overload
    def add_edge_attribute(
        self,
        name: str,
        dtype: Literal["float64", "double"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> EdgeAttributeFloat64: ...
    @overload
    def add_edge_attribute(
        self,
        name: str,
        dtype: Literal["int32", "int"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> EdgeAttributeInt32: ...
    @overload
    def add_edge_attribute(
        self,
        name: str,
        dtype: Literal["int8"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> EdgeAttributeInt8: ...
    @overload
    def add_edge_attribute(
        self,
        name: str,
        dtype: _DTypeName,
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> Attribute: ...

    @overload
    def add_face_attribute(
        self,
        name: str,
        dtype: Literal["float32", "float"] = "float32",
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> FaceAttributeFloat32: ...
    @overload
    def add_face_attribute(
        self,
        name: str,
        dtype: Literal["float64", "double"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> FaceAttributeFloat64: ...
    @overload
    def add_face_attribute(
        self,
        name: str,
        dtype: Literal["int32", "int"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> FaceAttributeInt32: ...
    @overload
    def add_face_attribute(
        self,
        name: str,
        dtype: Literal["int8"],
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> FaceAttributeInt8: ...
    @overload
    def add_face_attribute(
        self,
        name: str,
        dtype: _DTypeName,
        dim: _IntLike = 1,
        location: _LocationArg = "all",
        layout: _LayoutArg = "soa",
    ) -> Attribute: ...

    def sparse_matrix(
        self,
        op: _OpArg = "vv",
        dtype: _NumericDType = "float32",
    ) -> SparseMatrix: ...


class CholeskySolver:
    def __init__(
        self,
        matrix: SparseMatrix,
        permute: _Permutation = "none",
    ) -> None: ...
    @property
    def permute(self) -> _Permutation: ...
    def pre_solve(self, mesh: RXMeshStatic) -> None: ...


class QRSolver:
    def __init__(
        self,
        matrix: SparseMatrix,
        permute: _Permutation = "none",
    ) -> None: ...
    @property
    def permute(self) -> _Permutation: ...
    def pre_solve(self, mesh: RXMeshStatic) -> None: ...


class LUSolver:
    def __init__(
        self,
        matrix: SparseMatrix,
        permute: _Permutation = "none",
    ) -> None: ...
    @property
    def permute(self) -> _Permutation: ...
    def pre_solve(self, mesh: RXMeshStatic) -> None: ...


class cuDSSCholeskySolver:
    def __init__(
        self,
        matrix: SparseMatrix,
        permute: _Permutation = "none",
    ) -> None: ...
    @property
    def permute(self) -> _Permutation: ...
    def pre_solve(
        self,
        mesh: RXMeshStatic,
        rhs: DenseMatrix,
        solution: DenseMatrix,
    ) -> None: ...


class ScalarEnergy:
    @property
    def dtype(self) -> _CanonicalFloatDType: ...
    @property
    def element_kind(self) -> _ElementName: ...
    @property
    def mesh(self) -> RXMeshStatic: ...
    def evaluate(
        self,
        opt_var: Attribute,
        stream: _Stream = None,
    ) -> float: ...
    def _torch_forward(
        self,
        input_ptr: _IntLike,
        gradient_ptr: _IntLike,
        term_losses_ptr: _IntLike,
        stream: _Stream,
    ) -> None: ...
    def gradient_snapshot(self, stream: _Stream = None) -> DenseMatrix: ...
    def torch(
        self,
        x: torch.Tensor,
        *,
        copy: _CopyPolicy = "auto",
    ) -> torch.Tensor: ...


@overload
def create_plane(
    nx: _IntLike,
    ny: _IntLike,
    plane: _IntLike = 1,
    dx: float = 1.0,
    with_cross_diagonal: bool = False,
    low_corner: Sequence[float] = (0.0, 0.0, 0.0),
    dtype: Literal["float32", "float"] = "float32",
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]: ...
@overload
def create_plane(
    nx: _IntLike,
    ny: _IntLike,
    plane: _IntLike = 1,
    dx: float = 1.0,
    with_cross_diagonal: bool = False,
    low_corner: Sequence[float] = (0.0, 0.0, 0.0),
    dtype: Literal["float64", "double"] = "float64",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]: ...


def cuda_stream_synchronize(stream: _Stream = None) -> None: ...


has_cudss: Final[bool]


class VertexHandle:
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...


class EdgeHandle:
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...


class FaceHandle:
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...


class DEdgeHandle:
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...


class RXMeshStatic:
    def patch_size_stats(self) -> tuple[int, int, int]: ...
    def scale(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None: ...


def init(device_id: _IntLike = 0, log_level: LogLevel = ...) -> None: ...

