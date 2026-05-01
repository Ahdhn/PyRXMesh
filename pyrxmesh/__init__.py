"""Python bindings for RXMesh."""

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
        DType,
        ElementKind,
        Layout,
        Location,
        Op,
        RXMeshStatic,
        abi_version,
        build_config_tag,
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
        "DType",
        "ElementKind",
        "Layout",
        "Location",
        "Op",
        "RXMeshStatic",
        "abi_version",
        "build_config_tag",
        "init",
        "show",
    ]


def __getattr__(name: str):
    if _extension_import_error is not None:
        raise ModuleNotFoundError(
            "pyrxmesh._rxmesh is not installed yet. Build/install PyRXMesh "
            "before using runtime mesh bindings."
        ) from _extension_import_error
    raise AttributeError(name)
