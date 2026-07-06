from pathlib import Path

import pytest

import pyrxmesh as rx


@pytest.fixture(scope="session", autouse=True)
def _rxmesh_init():
    try:
        rx.init()
    except RuntimeError as exc:
        if "logger with name 'RXMesh' already exists" not in str(exc):
            raise


_MESH_PATH = Path(__file__).resolve().parents[1] / "meshes" / "sphere3.obj"


@pytest.fixture()
def mesh_path() -> Path:
    assert _MESH_PATH.exists(), f"Missing test mesh: {_MESH_PATH}"
    return _MESH_PATH


@pytest.fixture()
def mesh(mesh_path: Path) -> rx.RXMeshStatic:
    return rx.RXMeshStatic(str(mesh_path), patch_size=256)
