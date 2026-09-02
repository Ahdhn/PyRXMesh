"""
Python bindings for RXMesh
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import torch
import typing
import typing_extensions
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias, overload
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from ._typing import BlockShape as _BlockShape, CanonicalDType as _CanonicalDType, CanonicalDenseOrder as _CanonicalDenseOrder, CanonicalFloatDType as _CanonicalFloatDType, CanonicalNumericDType as _CanonicalNumericDType, CopyPolicy as _CopyPolicy, DenseOrder as _DenseOrder, DTypeName as _DTypeName, ElementName as _ElementName, FloatDType as _FloatDType, IntLike as _IntLike, LayoutArg as _LayoutArg, LayoutName as _LayoutName, LocationArg as _LocationArg, LocationName as _LocationName, MeshOrder as _MeshOrder, NumericDType as _NumericDType, OpArg as _OpArg, OpResultName as _OpResultName, Permutation as _Permutation, Scalar as _Scalar, Shape2D as _Shape2D, Stream as _Stream
_Array: TypeAlias = npt.NDArray[Any]
_FloatArray: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]
_IndexArray: TypeAlias = npt.NDArray[np.uint32]
_HandleArray: TypeAlias = npt.NDArray[np.uint64]
_SparseArrays: TypeAlias = tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], _Array]
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
_JacobianResult: TypeAlias = _JacobianSparseMatrixFloat32 | _JacobianSparseMatrixFloat64
_HessianResult: TypeAlias = _HessianSparseMatrixFloat32Dim1 | _HessianSparseMatrixFloat32Dim2 | _HessianSparseMatrixFloat32Dim3 | _HessianSparseMatrixFloat32Dim4 | _HessianSparseMatrixFloat32Dim6 | _HessianSparseMatrixFloat64Dim1 | _HessianSparseMatrixFloat64Dim2 | _HessianSparseMatrixFloat64Dim3 | _HessianSparseMatrixFloat64Dim4 | _HessianSparseMatrixFloat64Dim6
__all__: list[str] = ['Attribute', 'CGSolver', 'CholeskySolver', 'DEdgeHandle', 'DType', 'DenseMatrix', 'EdgeAttributeFloat32', 'EdgeAttributeFloat64', 'EdgeAttributeInt32', 'EdgeAttributeInt8', 'EdgeHandle', 'ElementKind', 'FaceAttributeFloat32', 'FaceAttributeFloat64', 'FaceAttributeInt32', 'FaceAttributeInt8', 'FaceHandle', 'HessianSparseMatrix', 'JacobianSparseMatrix', 'LUSolver', 'Layout', 'Location', 'LogLevel', 'Op', 'PCGSolver', 'QRSolver', 'RXMeshStatic', 'ScalarEnergy', 'SparseMatrix', 'VertexAttributeFloat32', 'VertexAttributeFloat64', 'VertexAttributeInt32', 'VertexAttributeInt8', 'VertexHandle', 'abi_version', 'build_config_tag', 'create_plane', 'cuDSSCholeskySolver', 'cuda_stream_synchronize', 'has_cudss', 'init', 'show']

class Attribute:

    def __dlpack__(self, stream: _Stream=None) -> object:
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        ...

    def __rxmesh_capsule__(self) -> typing_extensions.CapsuleType:
        """
        Return a low-level capsule for compiled PyRXMesh plugins.
        """

    def argmax(self, column: _IntLike | None=None) -> tuple[_BasicHandle, _Scalar]:
        ...

    def argmin(self, column: _IntLike | None=None) -> tuple[_BasicHandle, _Scalar]:
        ...

    def copy_from(self, other: Attribute, source: _LocationArg='all', target: _LocationArg='all') -> None:
        ...

    def dot(self, other: Attribute, column: typing.SupportsInt | typing.SupportsIndex | None=None) -> typing.Any:
        ...

    def from_dlpack_copy(self, source: object, target: _LocationArg='all') -> None:
        ...

    def from_matrix_copy(self, matrix: DenseMatrix, target: _LocationArg='all') -> None:
        ...

    def from_numpy_copy(self, values: npt.ArrayLike, target: _LocationArg='all') -> None:
        ...

    def from_torch_copy(self, values: torch.Tensor, target: _LocationArg='all') -> Attribute:
        ...

    def move(self, source: _LocationArg, target: _LocationArg) -> None:
        ...

    def norm2(self, column: typing.SupportsInt | typing.SupportsIndex | None=None) -> typing.Any:
        ...

    def reduce_max(self, column: typing.SupportsInt | typing.SupportsIndex | None=None) -> typing.Any:
        ...

    def reduce_min(self, column: typing.SupportsInt | typing.SupportsIndex | None=None) -> typing.Any:
        ...

    def reduce_sum(self, column: typing.SupportsInt | typing.SupportsIndex | None=None) -> typing.Any:
        ...

    def reset(self, value: _Scalar, location: _LocationArg='all') -> None:
        ...

    def to_dlpack(self, location: _LocationArg='device', stream: _Stream=None) -> object:
        ...

    def to_matrix_copy(self) -> DenseMatrix:
        ...

    def to_numpy(self, location: _LocationArg='host') -> _Array:
        ...

    def to_numpy_copy(self, source: _LocationArg='host') -> _Array:
        ...

    def to_torch(self, location: _LocationArg='device') -> torch.Tensor:
        ...

    @property
    def allocated(self) -> _LocationName:
        ...

    @property
    def bytes(self) -> int:
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def dtype(self) -> _CanonicalDType:
        ...

    @property
    def element_count(self) -> int:
        ...

    @property
    def element_kind(self) -> _ElementName:
        ...

    @property
    def is_device_allocated(self) -> bool:
        ...

    @property
    def is_host_allocated(self) -> bool:
        ...

    @property
    def is_tensor_layout(self) -> bool:
        ...

    @property
    def layout(self) -> _LayoutName:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def shape(self) -> _Shape2D:
        ...

    @property
    def size(self) -> int:
        ...

class CGSolver:

    def __init__(self, matrix: SparseMatrix, unknown_dim: typing.SupportsInt | typing.SupportsIndex=1, max_iter: typing.SupportsInt | typing.SupportsIndex=1000, abs_tol: typing.Any=1e-06, rel_tol: typing.Any=0.0, reset_residual_freq: typing.SupportsInt | typing.SupportsIndex=2147483647) -> None:
        ...

    def pre_solve(self, rhs: DenseMatrix, solution: DenseMatrix) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, solution: DenseMatrix, pre_solve: bool=True) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, initial_guess: typing.Any=None, pre_solve: bool=True) -> DenseMatrix:
        ...

    @property
    def final_residual(self) -> typing.Any:
        ...

    @property
    def iter_taken(self) -> int:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def start_residual(self) -> typing.Any:
        ...

class CholeskySolver:

    def __init__(self, matrix: SparseMatrix, permute: _Permutation='none') -> None:
        ...

    def pre_solve(self, mesh: RXMeshStatic) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, solution: DenseMatrix, pre_solve: bool=False) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, initial_guess: typing.Any=None, pre_solve: bool=False) -> DenseMatrix:
        ...

    @property
    def is_factorized(self) -> bool:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def permute(self) -> _Permutation:
        ...

class DEdgeHandle:

    def __eq__(self, other: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    @typing.overload
    def __init__(self) -> None:
        ...

    @typing.overload
    def __init__(self, patch_id: typing.SupportsInt | typing.SupportsIndex, local_id: typing.SupportsInt | typing.SupportsIndex, direction: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def edge_handle(self) -> EdgeHandle:
        ...

    def flipped(self) -> DEdgeHandle:
        ...

    def is_valid(self) -> bool:
        ...

    @property
    def local_id(self) -> int:
        ...

    @property
    def patch_id(self) -> int:
        ...

    @property
    def unique_id(self) -> int:
        ...

class DType:
    """
    Members:
    
      Float32
    
      Float64
    
      Int32
    
      Int8
    """
    Float32: typing.ClassVar[DType]
    Float64: typing.ClassVar[DType]
    Int32: typing.ClassVar[DType]
    Int8: typing.ClassVar[DType]
    __members__: typing.ClassVar[dict[str, DType]]

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...

class DenseMatrix:

    @staticmethod
    def from_dlpack_copy(source: object, order: _DenseOrder='col_major') -> DenseMatrix:
        ...

    @staticmethod
    def from_dlpack_view(source: object, order: _DenseOrder='col_major') -> DenseMatrix:
        ...

    @staticmethod
    def from_torch_copy(source: torch.Tensor, order: _DenseOrder='col_major') -> DenseMatrix:
        ...

    @staticmethod
    def from_torch_view(source: torch.Tensor, order: _DenseOrder='col_major') -> DenseMatrix:
        ...

    def __dlpack__(self, stream: _Stream=None) -> object:
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        ...

    @overload
    def __init__(self, rows: _IntLike, cols: _IntLike, dtype: _NumericDType='float32', location: _LocationArg='all', order: _DenseOrder='col_major') -> None:
        ...

    @overload
    def __init__(self, mesh: RXMeshStatic, rows: _IntLike, cols: _IntLike, dtype: _NumericDType='float32', location: _LocationArg='all', order: _DenseOrder='col_major') -> None:
        ...

    def abs_max(self, stream: typing.Any=None) -> typing.Any:
        ...

    def abs_min(self, stream: typing.Any=None) -> typing.Any:
        ...

    def abs_sum(self, stream: typing.Any=None) -> typing.Any:
        ...

    def axpy(self, x: DenseMatrix, alpha: typing.Any, stream: typing.Any=None) -> None:
        ...

    def col(self, column: typing.SupportsInt | typing.SupportsIndex) -> DenseMatrix:
        ...

    def copy_from(self, other: DenseMatrix, source: _LocationArg='all', target: _LocationArg='all', stream: _Stream=None) -> None:
        ...

    def dot(self, other: DenseMatrix, stream: typing.Any=None) -> typing.Any:
        ...

    def fill_random(self, low: typing.SupportsFloat | typing.SupportsIndex=-1.0, high: typing.SupportsFloat | typing.SupportsIndex=1.0) -> None:
        ...

    def from_numpy_copy(self, values: npt.ArrayLike, target: _LocationArg='all', stream: _Stream=None) -> None:
        ...

    def move(self, source: _LocationArg, target: _LocationArg, stream: _Stream=None) -> None:
        ...

    def multiply(self, scalar: typing.Any, stream: typing.Any=None) -> None:
        ...

    def norm2(self, stream: typing.Any=None) -> typing.Any:
        ...

    def release(self, location: _LocationArg='all') -> None:
        ...

    def reset(self, value: _Scalar, location: _LocationArg='all', stream: _Stream=None) -> None:
        ...

    def reshape(self, rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def segment(self, start: typing.SupportsInt | typing.SupportsIndex, count: typing.SupportsInt | typing.SupportsIndex) -> DenseMatrix:
        ...

    def set_value(self, row_or_handle: typing.Any, col: typing.SupportsInt | typing.SupportsIndex, value: typing.Any) -> None:
        ...

    def swap(self, other: DenseMatrix, stream: typing.Any=None) -> None:
        ...

    def to_dlpack(self, location: _LocationArg='device', stream: _Stream=None) -> object:
        ...

    def to_mtx(self, file_name: str) -> None:
        ...

    def to_numpy(self, location: _LocationArg='host') -> _Array:
        ...

    def to_numpy_copy(self, source: _LocationArg='host') -> _Array:
        ...

    def to_torch(self, location: _LocationArg='device') -> torch.Tensor:
        ...

    def value(self, row_or_handle: typing.Any, col: typing.SupportsInt | typing.SupportsIndex=0) -> typing.Any:
        ...

    @property
    def bytes(self) -> int:
        ...

    @property
    def cols(self) -> int:
        ...

    @property
    def dtype(self) -> _CanonicalNumericDType:
        ...

    @property
    def is_device_allocated(self) -> bool:
        ...

    @property
    def is_host_allocated(self) -> bool:
        ...

    @property
    def is_read_only(self) -> bool:
        ...

    @property
    def is_view(self) -> bool:
        ...

    @property
    def location(self) -> _LocationName:
        ...

    @property
    def order(self) -> _CanonicalDenseOrder:
        ...

    @property
    def rows(self) -> int:
        ...

    @property
    def shape(self) -> _Shape2D:
        ...

class EdgeAttributeFloat32(Attribute):
    pass

class EdgeAttributeFloat64(Attribute):
    pass

class EdgeAttributeInt32(Attribute):
    pass

class EdgeAttributeInt8(Attribute):
    pass

class EdgeHandle:

    def __eq__(self, other: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    @typing.overload
    def __init__(self) -> None:
        ...

    @typing.overload
    def __init__(self, unique_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    @typing.overload
    def __init__(self, patch_id: typing.SupportsInt | typing.SupportsIndex, local_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def is_valid(self) -> bool:
        ...

    @property
    def local_id(self) -> int:
        ...

    @property
    def patch_id(self) -> int:
        ...

    @property
    def unique_id(self) -> int:
        ...

class ElementKind:
    """
    Members:
    
      Vertex
    
      Edge
    
      Face
    """
    Edge: typing.ClassVar[ElementKind]
    Face: typing.ClassVar[ElementKind]
    Vertex: typing.ClassVar[ElementKind]
    __members__: typing.ClassVar[dict[str, ElementKind]]

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...

class FaceAttributeFloat32(Attribute):
    pass

class FaceAttributeFloat64(Attribute):
    pass

class FaceAttributeInt32(Attribute):
    pass

class FaceAttributeInt8(Attribute):
    pass

class FaceHandle:

    def __eq__(self, other: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    @typing.overload
    def __init__(self) -> None:
        ...

    @typing.overload
    def __init__(self, unique_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    @typing.overload
    def __init__(self, patch_id: typing.SupportsInt | typing.SupportsIndex, local_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def is_valid(self) -> bool:
        ...

    @property
    def local_id(self) -> int:
        ...

    @property
    def patch_id(self) -> int:
        ...

    @property
    def unique_id(self) -> int:
        ...

class LUSolver:

    def __init__(self, matrix: SparseMatrix, permute: _Permutation='none') -> None:
        ...

    def pre_solve(self, mesh: RXMeshStatic) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, solution: DenseMatrix, pre_solve: bool=False) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, initial_guess: typing.Any=None, pre_solve: bool=False) -> DenseMatrix:
        ...

    @property
    def is_factorized(self) -> bool:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def permute(self) -> _Permutation:
        ...

class LogLevel:
    """
    Members:
    
      TRACE
    
      DEBUG
    
      INFO
    
      WARN
    
      ERROR
    
      CRITICAL
    
      OFF
    """
    CRITICAL: typing.ClassVar[LogLevel]
    DEBUG: typing.ClassVar[LogLevel]
    ERROR: typing.ClassVar[LogLevel]
    INFO: typing.ClassVar[LogLevel]
    OFF: typing.ClassVar[LogLevel]
    TRACE: typing.ClassVar[LogLevel]
    WARN: typing.ClassVar[LogLevel]
    __members__: typing.ClassVar[dict[str, LogLevel]]

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...

class Op:
    """
    Members:
    
      INVALID
    
      V
    
      E
    
      F
    
      VV
    
      VE
    
      VF
    
      FV
    
      FE
    
      FF
    
      EV
    
      EE
    
      EF
    
      EVDiamond
    """
    E: typing.ClassVar[Op]
    EE: typing.ClassVar[Op]
    EF: typing.ClassVar[Op]
    EV: typing.ClassVar[Op]
    EVDiamond: typing.ClassVar[Op]
    F: typing.ClassVar[Op]
    FE: typing.ClassVar[Op]
    FF: typing.ClassVar[Op]
    FV: typing.ClassVar[Op]
    INVALID: typing.ClassVar[Op]
    V: typing.ClassVar[Op]
    VE: typing.ClassVar[Op]
    VF: typing.ClassVar[Op]
    VV: typing.ClassVar[Op]
    __members__: typing.ClassVar[dict[str, Op]]

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...

class PCGSolver:

    def __init__(self, matrix: SparseMatrix, unknown_dim: typing.SupportsInt | typing.SupportsIndex=1, max_iter: typing.SupportsInt | typing.SupportsIndex=1000, abs_tol: typing.Any=1e-06, rel_tol: typing.Any=0.0, reset_residual_freq: typing.SupportsInt | typing.SupportsIndex=2147483647) -> None:
        ...

    def pre_solve(self, rhs: DenseMatrix, solution: DenseMatrix) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, solution: DenseMatrix, pre_solve: bool=True) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, initial_guess: typing.Any=None, pre_solve: bool=True) -> DenseMatrix:
        ...

    @property
    def final_residual(self) -> typing.Any:
        ...

    @property
    def iter_taken(self) -> int:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def start_residual(self) -> typing.Any:
        ...

class QRSolver:

    def __init__(self, matrix: SparseMatrix, permute: _Permutation='none') -> None:
        ...

    def pre_solve(self, mesh: RXMeshStatic) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, solution: DenseMatrix, pre_solve: bool=False) -> None:
        ...

    @typing.overload
    def solve(self, rhs: DenseMatrix, initial_guess: typing.Any=None, pre_solve: bool=False) -> DenseMatrix:
        ...

    @property
    def is_factorized(self) -> bool:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def permute(self) -> _Permutation:
        ...

class RXMeshStatic:

    @staticmethod
    def from_files(file_paths: collections.abc.Sequence[str], patch_size: typing.SupportsInt | typing.SupportsIndex=512) -> RXMeshStatic:
        """
        Load multiple OBJ files into one RXMeshStatic.
        """

    @staticmethod
    def show() -> None:
        """
        Open the Polyscope viewer for this mesh.
        """

    def __init__(self, file_path: str, patcher_file: str='', patch_size: typing.SupportsInt | typing.SupportsIndex=512, capacity_factor: typing.SupportsFloat | typing.SupportsIndex=1.0, patch_alloc_factor: typing.SupportsFloat | typing.SupportsIndex=1.0, lp_hashtable_load_factor: typing.SupportsFloat | typing.SupportsIndex=0.800000011920929) -> None:
        """
        Load a static triangle mesh from an OBJ file.
        """

    def __rxmesh_capsule__(self) -> typing_extensions.CapsuleType:
        """
        Return a low-level capsule for compiled PyRXMesh plugins.
        """

    def add_attribute_like(self, name: str, other: Attribute) -> typing.Any:
        """
        Add a new attribute with the same element kind, dtype, dimension, allocation, and layout as another attribute.
        """

    @overload
    def add_edge_attribute(self, name: str, dtype: Literal['float32', 'float']='float32', dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> EdgeAttributeFloat32:
        ...

    @overload
    def add_edge_attribute(self, name: str, dtype: Literal['float64', 'double'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> EdgeAttributeFloat64:
        ...

    @overload
    def add_edge_attribute(self, name: str, dtype: Literal['int32', 'int'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> EdgeAttributeInt32:
        ...

    @overload
    def add_edge_attribute(self, name: str, dtype: Literal['int8'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> EdgeAttributeInt8:
        ...

    @overload
    def add_edge_attribute(self, name: str, dtype: _DTypeName, dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> Attribute:
        ...

    @overload
    def add_face_attribute(self, name: str, dtype: Literal['float32', 'float']='float32', dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> FaceAttributeFloat32:
        ...

    @overload
    def add_face_attribute(self, name: str, dtype: Literal['float64', 'double'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> FaceAttributeFloat64:
        ...

    @overload
    def add_face_attribute(self, name: str, dtype: Literal['int32', 'int'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> FaceAttributeInt32:
        ...

    @overload
    def add_face_attribute(self, name: str, dtype: Literal['int8'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> FaceAttributeInt8:
        ...

    @overload
    def add_face_attribute(self, name: str, dtype: _DTypeName, dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> Attribute:
        ...

    @overload
    def add_vertex_attribute(self, name: str, dtype: Literal['float32', 'float']='float32', dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> VertexAttributeFloat32:
        ...

    @overload
    def add_vertex_attribute(self, name: str, dtype: Literal['float64', 'double'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> VertexAttributeFloat64:
        ...

    @overload
    def add_vertex_attribute(self, name: str, dtype: Literal['int32', 'int'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> VertexAttributeInt32:
        ...

    @overload
    def add_vertex_attribute(self, name: str, dtype: Literal['int8'], dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> VertexAttributeInt8:
        ...

    @overload
    def add_vertex_attribute(self, name: str, dtype: _DTypeName, dim: _IntLike=1, location: _LocationArg='all', layout: _LayoutArg='soa') -> Attribute:
        ...

    def attribute_names(self) -> list[str]:
        """
        Return names of attributes currently registered on this mesh.
        """

    def bounding_box(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        ...

    def edge_handles(self) -> _HandleArray:
        ...

    def export_obj(self, filename: str, coords: Attribute) -> None:
        """
        Export the mesh to an OBJ file using a vertex coordinate attribute.
        """

    def face_handles(self) -> _HandleArray:
        ...

    def faces(self, order: _MeshOrder='linear') -> npt.NDArray[np.uint32]:
        ...

    def for_each_edge(self, callback: Callable[[EdgeHandle], object]) -> None:
        ...

    def for_each_face(self, callback: Callable[[FaceHandle], object]) -> None:
        ...

    def for_each_vertex(self, callback: Callable[[VertexHandle], object]) -> None:
        ...

    @typing.overload
    def global_id(self, handle: VertexHandle) -> int:
        ...

    @typing.overload
    def global_id(self, handle: EdgeHandle) -> int:
        ...

    @typing.overload
    def global_id(self, handle: FaceHandle) -> int:
        ...

    def global_to_linear(self, element_kind: ElementKind) -> _IndexArray:
        ...

    def has_attribute(self, name: str) -> bool:
        """
        Return True when the mesh has an attribute with this name.
        """

    def input_vertex_coordinates(self) -> VertexAttributeFloat32 | VertexAttributeFloat64:
        ...

    def is_closed(self) -> bool:
        ...

    def is_edge_manifold(self) -> bool:
        ...

    @typing.overload
    def linear_id(self, handle: VertexHandle) -> int:
        ...

    @typing.overload
    def linear_id(self, handle: EdgeHandle) -> int:
        ...

    @typing.overload
    def linear_id(self, handle: FaceHandle) -> int:
        ...

    def linear_to_global(self, element_kind: ElementKind) -> _IndexArray:
        ...

    def patch_size_stats(self) -> tuple[int, int, int]:
        ...

    def remove_attribute(self, name: str) -> None:
        """
        Remove an attribute by name.
        """

    def save_patcher(self, filename: str) -> None:
        """
        Save RXMesh patching data to a file.
        """

    def scale(self, lower: Sequence[float], upper: Sequence[float]) -> None:
        ...

    def sparse_matrix(self, op: _OpArg='vv', dtype: _NumericDType='float32') -> SparseMatrix:
        ...

    def vertex_handles(self) -> _HandleArray:
        ...

    def vertices(self, order: _MeshOrder='linear') -> _FloatArray:
        ...

    @property
    def input_max_edge_incident_faces(self) -> int:
        ...

    @property
    def input_max_face_adjacent_faces(self) -> int:
        ...

    @property
    def input_max_valence(self) -> int:
        ...

    @property
    def max_num_patches(self) -> int:
        ...

    @property
    def num_colors(self) -> int:
        ...

    @property
    def num_components(self) -> int:
        ...

    @property
    def num_edges(self) -> int:
        ...

    @property
    def num_faces(self) -> int:
        ...

    @property
    def num_patches(self) -> int:
        ...

    @property
    def num_vertices(self) -> int:
        ...

    @property
    def patch_size(self) -> int:
        ...

    @property
    def patching_time(self) -> float:
        ...

class ScalarEnergy:

    def _torch_forward(self, input_ptr: _IntLike, gradient_ptr: _IntLike, term_losses_ptr: _IntLike, stream: _Stream) -> None:
        ...

    def evaluate(self, opt_var: Attribute, stream: _Stream=None) -> float:
        ...

    def gradient_snapshot(self, stream: _Stream=None) -> DenseMatrix:
        ...

    def torch(self, x: torch.Tensor, *, copy: _CopyPolicy='auto') -> torch.Tensor:
        ...

    @property
    def dtype(self) -> _CanonicalFloatDType:
        ...

    @property
    def element_count(self) -> int:
        ...

    @property
    def element_kind(self) -> _ElementName:
        ...

    @property
    def gradient_view(self) -> DenseMatrix:
        ...

    @property
    def has_evaluated(self) -> bool:
        ...

    @property
    def loss(self) -> float:
        ...

    @property
    def mesh(self) -> RXMeshStatic:
        ...

    @property
    def term_count(self) -> int:
        ...

    @property
    def variable_dim(self) -> int:
        ...

class SparseMatrix:

    @staticmethod
    def from_dlpack_copy(row_ptr: object, col_idx: object, values: object, shape: _Shape2D, dtype: _NumericDType | Literal['']='', stream: _Stream=None) -> SparseMatrix:
        ...

    @staticmethod
    def from_numpy_copy(row_ptr: npt.ArrayLike, col_idx: npt.ArrayLike, values: npt.ArrayLike, shape: _Shape2D, dtype: _NumericDType='float32') -> SparseMatrix:
        ...

    @staticmethod
    def from_torch_copy(source: torch.Tensor, dtype: _NumericDType | None=None, stream: _Stream=None) -> SparseMatrix:
        ...

    def __init__(self, mesh: RXMeshStatic, op: _OpArg='vv', dtype: _NumericDType='float32') -> None:
        ...

    def _col_indices_dlpack(self, location: typing.Any='device', stream: typing.Any=None) -> typing_extensions.CapsuleType:
        ...

    def _row_ptr_dlpack(self, location: typing.Any='device', stream: typing.Any=None) -> typing_extensions.CapsuleType:
        ...

    def _values_dlpack(self, location: typing.Any='device', stream: typing.Any=None) -> typing_extensions.CapsuleType:
        ...

    def copy_from(self, other: SparseMatrix, source: _LocationArg='all', target: _LocationArg='all', stream: _Stream=None) -> None:
        ...

    def from_dlpack_values_copy(self, values: object, target: _LocationArg='all', stream: _Stream=None) -> None:
        ...

    def from_numpy_values_copy(self, values: npt.ArrayLike, target: _LocationArg='all', stream: _Stream=None) -> None:
        ...

    def from_torch_values_copy(self, values: torch.Tensor, target: _LocationArg='all', stream: _Stream=None) -> SparseMatrix:
        ...

    def is_non_zero(self, row: typing.SupportsInt | typing.SupportsIndex, col: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...

    def move(self, source: _LocationArg, target: _LocationArg, stream: _Stream=None) -> None:
        ...

    def multiply(self, rhs: DenseMatrix, transpose_a: bool=False, transpose_b: bool=False, alpha: typing.Any=1.0, beta: typing.Any=0.0, stream: typing.Any=None) -> DenseMatrix:
        ...

    def multiply_vector(self, vector: DenseMatrix | npt.ArrayLike, stream: _Stream=None) -> DenseMatrix:
        ...

    def release(self) -> None:
        ...

    def reset(self, value: _Scalar, location: _LocationArg='all', stream: _Stream=None) -> None:
        ...

    def set_value(self, row: typing.SupportsInt | typing.SupportsIndex, col: typing.SupportsInt | typing.SupportsIndex, value: typing.Any) -> None:
        ...

    def sync_device_to_host(self, stream: typing.Any=None) -> None:
        ...

    def sync_host_to_device(self, stream: typing.Any=None) -> None:
        ...

    def to_dlpack(self, location: _LocationArg='device', stream: _Stream=None) -> tuple[object, object, object]:
        ...

    def to_file(self, file_name: str) -> None:
        ...

    def to_mtx(self, file_name: str) -> None:
        ...

    def to_numpy(self, location: _LocationArg='host') -> _SparseArrays:
        ...

    def to_numpy_copy(self, source: _LocationArg='host', stream: _Stream=None) -> _SparseArrays:
        ...

    def to_scipy_csr(self) -> csr_matrix:
        ...

    def to_scipy_csr_copy(self, source: _LocationArg='host', stream: _Stream=None) -> csr_matrix:
        ...

    def to_torch(self, location: _LocationArg='device') -> torch.Tensor:
        ...

    def value(self, row: typing.SupportsInt | typing.SupportsIndex, col: typing.SupportsInt | typing.SupportsIndex) -> typing.Any:
        ...

    def values_to_numpy(self, location: typing.Any='host') -> numpy.ndarray:
        ...

    def values_to_numpy_copy(self, source: typing.Any='host', stream: typing.Any=None) -> numpy.ndarray:
        ...

    @property
    def cols(self) -> int:
        ...

    @property
    def dtype(self) -> _CanonicalNumericDType:
        ...

    @property
    def index_dtype(self) -> Literal['int32']:
        ...

    @property
    def is_device_allocated(self) -> bool:
        ...

    @property
    def is_host_allocated(self) -> bool:
        ...

    @property
    def location(self) -> _LocationName:
        ...

    @property
    def lower_nnz(self) -> int:
        ...

    @property
    def nnz(self) -> int:
        ...

    @property
    def op(self) -> _OpResultName:
        ...

    @property
    def rows(self) -> int:
        ...

    @property
    def shape(self) -> _Shape2D:
        ...

class VertexAttributeFloat32(Attribute):
    pass

class VertexAttributeFloat64(Attribute):
    pass

class VertexAttributeInt32(Attribute):
    pass

class VertexAttributeInt8(Attribute):
    pass

class VertexHandle:

    def __eq__(self, other: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    @typing.overload
    def __init__(self) -> None:
        ...

    @typing.overload
    def __init__(self, unique_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    @typing.overload
    def __init__(self, patch_id: typing.SupportsInt | typing.SupportsIndex, local_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def is_valid(self) -> bool:
        ...

    @property
    def local_id(self) -> int:
        ...

    @property
    def patch_id(self) -> int:
        ...

    @property
    def unique_id(self) -> int:
        ...

class _HessianSparseMatrixFloat32Dim1(_SparseMatrixFloat32):

    @property
    def variable_dim(self) -> Literal[1]:
        ...

class _HessianSparseMatrixFloat32Dim2(_SparseMatrixFloat32):

    @property
    def variable_dim(self) -> Literal[2]:
        ...

class _HessianSparseMatrixFloat32Dim3(_SparseMatrixFloat32):

    @property
    def variable_dim(self) -> Literal[3]:
        ...

class _HessianSparseMatrixFloat32Dim4(_SparseMatrixFloat32):

    @property
    def variable_dim(self) -> Literal[4]:
        ...

class _HessianSparseMatrixFloat32Dim6(_SparseMatrixFloat32):

    @property
    def variable_dim(self) -> Literal[6]:
        ...

class _HessianSparseMatrixFloat64Dim1(_SparseMatrixFloat64):

    @property
    def variable_dim(self) -> Literal[1]:
        ...

class _HessianSparseMatrixFloat64Dim2(_SparseMatrixFloat64):

    @property
    def variable_dim(self) -> Literal[2]:
        ...

class _HessianSparseMatrixFloat64Dim3(_SparseMatrixFloat64):

    @property
    def variable_dim(self) -> Literal[3]:
        ...

class _HessianSparseMatrixFloat64Dim4(_SparseMatrixFloat64):

    @property
    def variable_dim(self) -> Literal[4]:
        ...

class _HessianSparseMatrixFloat64Dim6(_SparseMatrixFloat64):

    @property
    def variable_dim(self) -> Literal[6]:
        ...

class _JacobianSparseMatrixFloat32(_SparseMatrixFloat32):

    def term_num_rows(self, term: _IntLike) -> int:
        ...

    def term_rows_range(self, term: _IntLike) -> tuple[int, int]:
        ...

    @property
    def num_terms(self) -> int:
        ...

class _JacobianSparseMatrixFloat64(_SparseMatrixFloat64):

    def term_num_rows(self, term: _IntLike) -> int:
        ...

    def term_rows_range(self, term: _IntLike) -> tuple[int, int]:
        ...

    @property
    def num_terms(self) -> int:
        ...

class _SparseMatrixFloat32(SparseMatrix):
    pass

class _SparseMatrixFloat64(SparseMatrix):
    pass

class _SparseMatrixInt32(SparseMatrix):
    pass

class cuDSSCholeskySolver:

    def __init__(self, matrix: SparseMatrix, permute: _Permutation='none') -> None:
        ...

    @property
    def permute(self) -> _Permutation:
        ...

    def pre_solve(self, mesh: RXMeshStatic, rhs: DenseMatrix, solution: DenseMatrix) -> None:
        ...

def HessianSparseMatrix(mesh: RXMeshStatic, variable_dim: _IntLike=3, extra_nnz_entries: _IntLike=0, op: _OpArg='vv', dtype: _FloatDType='float32') -> _HessianResult:
    ...

def JacobianSparseMatrix(mesh: RXMeshStatic, ops: Sequence[_OpArg], block_shapes: Sequence[_BlockShape], dtype: _FloatDType='float32') -> _JacobianResult:
    ...

def abi_version() -> int:
    """
    Return the PyRXMesh plugin ABI version.
    """

def build_config_tag() -> str:
    """
    Return the RXMesh/PyRXMesh build configuration tag.
    """

@overload
def create_plane(nx: _IntLike, ny: _IntLike, plane: _IntLike=1, dx: float=1.0, with_cross_diagonal: bool=False, low_corner: Sequence[float]=(0.0, 0.0, 0.0), dtype: Literal['float32', 'float']='float32') -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    ...

@overload
def create_plane(nx: _IntLike, ny: _IntLike, plane: _IntLike=1, dx: float=1.0, with_cross_diagonal: bool=False, low_corner: Sequence[float]=(0.0, 0.0, 0.0), dtype: Literal['float64', 'double']='float64') -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    ...

def cuda_stream_synchronize(stream: _Stream=None) -> None:
    ...

def init(device_id: _IntLike=0, log_level: LogLevel=...) -> None:
    ...

def show() -> None:
    """
    Open the Polyscope viewer.
    """
has_cudss: Final[bool]
