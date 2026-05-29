from __future__ import annotations

from pathlib import Path

import numpy as np
import pyrxmesh as rx
import pytest
import scipy.sparse as sp


try:
    rx.init()
except RuntimeError as exc:
    if "logger with name 'RXMesh' already exists" not in str(exc):
        raise


def load_mesh() -> rx.RXMeshStatic:
    mesh_path = Path(__file__).resolve().parents[1] / "meshes" / "sphere3.obj"
    assert mesh_path.exists(), f"Missing test mesh: {mesh_path}"
    return rx.RXMeshStatic(str(mesh_path), patch_size=256)


def test_geometry_create_plane() -> None:    
    vertices, faces = rx.create_plane(
        4,
        3,
        plane=2,
        dx=0.5,
        with_cross_diagonal=True,
        low_corner=(1.0, 2.0, 3.0),
    )

    assert vertices.shape == (12, 3)
    assert faces.shape == (18, 3)
    assert vertices.dtype == np.float32
    assert faces.dtype == np.uint32
    np.testing.assert_allclose(vertices[0], [1.0, 2.0, 3.0])
    assert np.all(faces < len(vertices))


def test_io_export_obj(tmp_path: Path) -> None:
    mesh = load_mesh()
    coords = mesh.input_vertex_coordinates()

    output_path = tmp_path / "sphere.obj"    
    mesh.export_obj(str(output_path), coords)

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "\nv " in f"\n{text}"
    assert "\nf " in f"\n{text}"


def test_io_export_obj_rejects_non_vertex_coords(tmp_path: Path) -> None:    
    mesh = load_mesh()
    edge_attr = mesh.add_edge_attribute("bad_coords", dtype="float32", dim=3)

    with pytest.raises(ValueError):
        mesh.export_obj(str(tmp_path / "bad.obj"), edge_attr)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
