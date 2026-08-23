"""Build and import external example plugins against an installed wheel."""

from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _copy_plugin(
    project_root: Path, destination: Path, example_name: str
) -> Path:
    source = project_root / "examples" / example_name
    if not (source / "pyproject.toml").is_file():
        raise RuntimeError(f"Missing external plugin source: {source}")

    plugin_root = destination / example_name
    shutil.copytree(
        source,
        plugin_root,
        ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc"),
    )
    return plugin_root


def _import_installed_module(module_name: str, project_root: Path):
    """Import an installed module without resolving the source checkout."""

    project_root = project_root.resolve()
    original_path = sys.path[:]
    try:
        sys.path[:] = [
            entry
            for entry in original_path
            if not (
                Path(entry or Path.cwd()).resolve() == project_root
                or project_root
                in Path(entry or Path.cwd()).resolve().parents
            )
        ]
        module = importlib.import_module(module_name)
    finally:
        sys.path[:] = original_path

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(
            f"Wheel verification imported {module_name} without a module file"
        )
    module_path = Path(module_file).resolve()
    if module_path == project_root or project_root in module_path.parents:
        raise RuntimeError(
            f"Wheel verification imported {module_name} from the source "
            f"checkout ({module_path}) instead of site-packages"
        )
    return module


def _import_installed_pyrxmesh(project_root: Path):
    """Import the installed PyRXMesh package with the source-tree guard."""

    return _import_installed_module("pyrxmesh", project_root)


def _plugin_cuda_architecture(build_config: str) -> str:
    """Select one device-link-compatible architecture from a runtime tag."""

    marker = ",cuda_arch="
    if marker not in build_config:
        raise RuntimeError(
            "Installed PyRXMesh build_config_tag() has no cuda_arch field: "
            f"{build_config!r}"
        )

    value = build_config.rsplit(marker, 1)[1]
    if value == "native":
        return value

    architectures = value.split(",")
    if not architectures or any(
        not architecture.isascii() or not architecture.isdecimal()
        for architecture in architectures
    ):
        raise RuntimeError(
            "Installed PyRXMesh has an unsupported cuda_arch value; expected "
            f"'native' or a comma-separated numeric list, got {value!r}"
        )
    return architectures[0]


def _install_plugin(plugin_root: Path, cuda_architecture: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-v",
            "--no-build-isolation",
            "--no-deps",
            "--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="
            f"{cuda_architecture}",
            str(plugin_root),
        ],
        check=True,
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_wheel_plugin.py PROJECT_ROOT")

    project_root = Path(sys.argv[1]).resolve()
    rx = _import_installed_pyrxmesh(project_root)
    abi = rx.abi_version()
    config = rx.build_config_tag()
    plugin_cuda_architecture = _plugin_cuda_architecture(config)

    with tempfile.TemporaryDirectory(prefix="pyrxmesh-wheel-plugin-") as tmp:
        temporary_root = Path(tmp)
        for example_name in ("diff_energy_plugin", "custom_kernel_plugin"):
            plugin_root = _copy_plugin(
                project_root, temporary_root, example_name
            )
            _install_plugin(plugin_root, plugin_cuda_architecture)

    importlib.invalidate_caches()
    diff_plugin = _import_installed_module(
        "rxmesh_diff_energy", project_root
    )
    kernel_plugin = _import_installed_module(
        "rxmesh_edge_lengths", project_root
    )

    if f"abi={abi}" not in config:
        raise RuntimeError(
            f"Runtime build tag does not identify ABI {abi}: {config!r}"
        )
    if not callable(diff_plugin.make_energy):
        raise RuntimeError("rxmesh_diff_energy.make_energy is not callable")
    if not callable(kernel_plugin.compute_edge_lengths):
        raise RuntimeError(
            "rxmesh_edge_lengths.compute_edge_lengths is not callable"
        )

    stale_abi = abi - 1
    try:
        diff_plugin._test_require_compatible_runtime(stale_abi, config)
    except RuntimeError as exc:
        message = str(exc)
        required_diagnostics = (
            f"Plugin ABI (compiled/expected): {stale_abi}",
            f"runtime ABI (active/actual): {abi}",
            "Rebuild the plugin in the active environment.",
        )
        missing = [
            diagnostic
            for diagnostic in required_diagnostics
            if diagnostic not in message
        ]
        if missing:
            raise RuntimeError(
                "Stale-plugin rejection was not actionable; missing "
                f"diagnostics: {missing}. Full message: {message!r}"
            ) from exc
    else:
        raise RuntimeError(
            f"Runtime accepted stale plugin ABI {stale_abi}; expected ABI {abi}"
        )

    print(
        "external example plugins imported successfully "
        f"(ABI {abi}, CUDA architecture {plugin_cuda_architecture})"
    )
    print("verified rxmesh_diff_energy.make_energy")
    print("verified rxmesh_edge_lengths.compute_edge_lengths")
    print(f"stale plugin ABI {stale_abi} rejected with actionable diagnostics")
    print(config)


if __name__ == "__main__":
    main()
