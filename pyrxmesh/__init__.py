"""PyRXMesh: Python bindings for the RXMesh GPU mesh-processing library.

This package exposes:

- The native classes (``RXMeshStatic``, ``DenseMatrix``, ``SparseMatrix``,
  ``Attribute``, solvers, etc.) from the compiled ``_rxmesh`` extension.
- Optional NumPy / SciPy / PyTorch interop methods attached to those classes
  via the ``_numpy_interop``, ``_scipy_interop``, and ``_torch_interop``
  submodules. The torch / scipy modules import their backing library lazily,
  so attaching them is free if the user never calls them.
"""

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
        cuda_stream_synchronize,
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
        "cuda_stream_synchronize",
        "cuDSSCholeskySolver",
        "has_cudss",
        "init",
        "show",
    ]

    # Attach optional interop methods to the native classes. Each submodule
    # only does the patching; backing libraries (torch, scipy) are imported
    # lazily inside the patched methods, so loading these modules is cheap.
    from . import _numpy_interop  # noqa: F401  (side-effect: monkey-patches)
    from . import _scipy_interop  # noqa: F401  (side-effect: monkey-patches)
    from . import _torch_interop  # noqa: F401  (side-effect: monkey-patches)


def __getattr__(name: str):
    if _extension_import_error is not None:
        raise ModuleNotFoundError(
            "pyrxmesh._rxmesh is not installed yet. Build/install PyRXMesh "
            "before using runtime mesh bindings."
        ) from _extension_import_error
    raise AttributeError(name)
