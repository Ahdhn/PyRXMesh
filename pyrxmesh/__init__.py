from __future__ import annotations

import os
from pathlib import Path


_dll_handles = []


def _add_windows_dll_dir(path: Path) -> None:
    if os.name != "nt" or not path.exists():
        return
    try:
        handle = os.add_dll_directory(str(path))
    except (FileNotFoundError, OSError):
        return
    _dll_handles.append(handle)


def _prepare_dll_search_path() -> None:
    if os.name != "nt":
        return

    package_dir = Path(__file__).resolve().parent
    _add_windows_dll_dir(package_dir)
    _add_windows_dll_dir(package_dir / "bin")
    _add_windows_dll_dir(package_dir / "lib")

    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        cuda_root = os.environ.get(env_name)
        if cuda_root:
            _add_windows_dll_dir(Path(cuda_root) / "bin")

    for env_name, cuda_root in os.environ.items():
        if env_name.startswith("CUDA_PATH_V") and cuda_root:
            _add_windows_dll_dir(Path(cuda_root) / "bin")

    default_cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if default_cuda_root.exists():
        for cuda_bin in sorted(default_cuda_root.glob("v*/bin"), reverse=True):
            _add_windows_dll_dir(cuda_bin)


_prepare_dll_search_path()

_extension_import_error: ModuleNotFoundError | None = None

try:
    from ._rxmesh import (
        Attribute,
        CGSolver,
        CandidatePairs,
        CholeskySolver,
        DEdgeHandle,
        DType,
        DenseMatrix,
        EdgeAttributeFloat32,
        EdgeAttributeFloat64,
        EdgeAttributeInt32,
        EdgeAttributeInt8,
        EdgeHandle,
        ElementKind,
        FaceAttributeFloat32,
        FaceAttributeFloat64,
        FaceAttributeInt32,
        FaceAttributeInt8,
        FaceHandle,
        HessianSparseMatrix,
        JacobianSparseMatrix,
        Layout,
        Location,
        LogLevel,
        Op,
        PCGSolver,
        QRSolver,
        RXMeshStatic,
        LUSolver,
        SparseMatrix,
        VertexAttributeFloat32,
        VertexAttributeFloat64,
        VertexAttributeInt32,
        VertexAttributeInt8,
        VertexHandle,
        abi_version,
        build_config_tag,
        create_plane,
        cuDSSCholeskySolver,
        has_cudss,
        init,
        show,
    )
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}._rxmesh":
        raise
    _extension_import_error = exc
    __all__ = []
else:
    __all__ = [
        "Attribute",
        "CGSolver",
        "CandidatePairs",
        "CholeskySolver",
        "DEdgeHandle",
        "DType",
        "DenseMatrix",
        "EdgeAttributeFloat32",
        "EdgeAttributeFloat64",
        "EdgeAttributeInt32",
        "EdgeAttributeInt8",
        "EdgeHandle",
        "ElementKind",
        "FaceAttributeFloat32",
        "FaceAttributeFloat64",
        "FaceAttributeInt32",
        "FaceAttributeInt8",
        "FaceHandle",
        "HessianSparseMatrix",
        "JacobianSparseMatrix",
        "Layout",
        "Location",
        "LogLevel",
        "Op",
        "PCGSolver",
        "QRSolver",
        "RXMeshStatic",
        "LUSolver",
        "SparseMatrix",
        "VertexAttributeFloat32",
        "VertexAttributeFloat64",
        "VertexAttributeInt32",
        "VertexAttributeInt8",
        "VertexHandle",
        "abi_version",
        "build_config_tag",
        "create_plane",
        "cuDSSCholeskySolver",
        "has_cudss",
        "init",
        "show",
    ]

    def _require_torch():
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for PyRXMesh torch interop helpers. "
                "Install torch or use the DLPack/NumPy APIs directly."
            ) from exc
        return torch

    class _DenseMatrixDlpackView:
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

    def _dense_matrix_to_torch(self, location=Location.DEVICE):
        torch = _require_torch()
        return torch.utils.dlpack.from_dlpack(
            _DenseMatrixDlpackView(self, location)
        )

    def _dense_matrix_from_torch_copy(source):
        tensor = source.detach() if hasattr(source, "detach") else source
        return DenseMatrix.from_dlpack_copy(tensor)

    def _sparse_matrix_to_torch(self, location=Location.DEVICE):
        torch = _require_torch()
        row_ptr, col_idx, values = self.to_dlpack_csr(location)
        crow = torch.utils.dlpack.from_dlpack(row_ptr)
        col = torch.utils.dlpack.from_dlpack(col_idx)
        val = torch.utils.dlpack.from_dlpack(values)
        return torch.sparse_csr_tensor(
            crow,
            col,
            val,
            size=self.shape,
            device=val.device,
        )

    def _sparse_matrix_from_torch_values_copy(self, values, target=Location.ALL):
        tensor = values.detach() if hasattr(values, "detach") else values
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        self.from_numpy_values(tensor.numpy(), target=target)
        return self

    def _torch_dtype_name(dtype):
        torch = _require_torch()
        if dtype == torch.float32:
            return "float32"
        if dtype == torch.float64:
            return "float64"
        if dtype == torch.int32:
            return "int32"
        raise TypeError(f"Unsupported torch dtype for PyRXMesh copy: {dtype}")

    def _sparse_matrix_from_torch_csr_copy(source, dtype=None):
        torch = _require_torch()
        if source.layout != torch.sparse_csr:
            raise TypeError("SparseMatrix.from_torch_csr_copy() expects a torch sparse CSR tensor.")
        value_dtype = dtype or _torch_dtype_name(source.values().dtype)
        return SparseMatrix.from_csr_copy(
            source.crow_indices().detach().cpu().numpy(),
            source.col_indices().detach().cpu().numpy(),
            source.values().detach().cpu().numpy(),
            tuple(source.shape),
            dtype=value_dtype,
        )

    def _attribute_to_torch(self, location=Location.DEVICE):
        torch = _require_torch()
        return torch.utils.dlpack.from_dlpack(self.to_dlpack(location))

    def _attribute_from_torch_copy(self, values, target=Location.ALL):
        tensor = values.detach() if hasattr(values, "detach") else values
        self.from_dlpack_copy(tensor, target=target)
        return self

    def _sparse_matrix_to_scipy_csr(self, source=Location.HOST, copy=True):
        try:
            import scipy.sparse as scipy_sparse
        except ImportError as exc:
            raise ImportError(
                "SciPy is required for SparseMatrix.to_scipy_csr(). "
                "Install scipy or use to_numpy_csr()."
            ) from exc

        row_ptr, col_idx, values = self.to_numpy_csr(source=source, copy=copy)
        return scipy_sparse.csr_matrix((values, col_idx, row_ptr), shape=self.shape)

    _native_sparse_matrix_multiply_vector = SparseMatrix.multiply_vector

    def _sparse_matrix_multiply_vector(self, vector, alpha=1.0, beta=0.0):
        if isinstance(vector, DenseMatrix):
            return _native_sparse_matrix_multiply_vector(self, vector, alpha, beta)

        import numpy as np

        values = np.asarray(vector)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(
                "SparseMatrix.multiply_vector() expects a 1D array or a "
                "DenseMatrix/array with shape (cols, 1)."
            )
        if values.shape[0] != self.cols:
            raise ValueError(
                "SparseMatrix.multiply_vector() vector length must match matrix.cols."
            )

        dense = DenseMatrix(
            self.cols,
            1,
            dtype=self.dtype,
            location=Location.ALL,
        )
        dense.from_numpy_copy(values, target=Location.ALL)
        return _native_sparse_matrix_multiply_vector(self, dense, alpha, beta)

    DenseMatrix.to_torch = _dense_matrix_to_torch
    DenseMatrix.from_torch_copy = staticmethod(_dense_matrix_from_torch_copy)
    SparseMatrix.to_torch = _sparse_matrix_to_torch
    SparseMatrix.from_torch_csr_copy = staticmethod(_sparse_matrix_from_torch_csr_copy)
    SparseMatrix.from_torch_values_copy = _sparse_matrix_from_torch_values_copy
    SparseMatrix.to_scipy_csr = _sparse_matrix_to_scipy_csr
    SparseMatrix.multiply_vector = _sparse_matrix_multiply_vector
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


def __getattr__(name: str):
    if _extension_import_error is not None:
        raise ModuleNotFoundError(
            "pyrxmesh._rxmesh is not installed yet. Build/install PyRXMesh "
            "before using runtime mesh bindings."
        ) from _extension_import_error
    raise AttributeError(name)
