from pathlib import Path
import numpy as np
import pyrxmesh as rx
import pytest

try:
    rx.init()
except RuntimeError as exc:
    if "logger with name 'RXMesh' already exists" not in str(exc):
        raise


def load_mesh() -> rx.RXMeshStatic:
    mesh_path = Path(__file__).resolve().parents[1] / "meshes" / "sphere3.obj"
    assert mesh_path.exists(), f"Missing test mesh: {mesh_path}"
    return rx.RXMeshStatic(str(mesh_path), patch_size=256)


def test_handle_round_trip_and_repr() -> None:
    vertex = rx.VertexHandle(7, 3)
    same = rx.VertexHandle(int(vertex))

    assert vertex.is_valid()
    assert vertex == same
    assert hash(vertex) == hash(same)
    assert vertex.patch_id == 7
    assert vertex.local_id == 3
    assert "VertexHandle" in repr(vertex)

    edge = rx.EdgeHandle(2, 5)
    face = rx.FaceHandle(1, 4)
    assert edge.patch_id == 2
    assert edge.local_id == 5
    assert face.patch_id == 1
    assert face.local_id == 4

    dedge = rx.DEdgeHandle(2, 5, 0)
    assert dedge.edge_handle() == edge
    assert dedge.flipped() != dedge


def test_core_mesh_metadata_and_topology_arrays() -> None:
    mesh = load_mesh()

    assert mesh.num_vertices > 0
    assert mesh.num_faces > 0
    assert mesh.num_edges > 0
    assert mesh.num_patches >= 1
    assert mesh.max_num_patches >= mesh.num_patches
    assert mesh.patch_size == 256
    assert mesh.input_max_valence > 0
    assert mesh.is_edge_manifold()

    vertices = mesh.vertices()
    faces = mesh.faces()

    assert vertices.shape == (mesh.num_vertices, 3)
    assert faces.shape == (mesh.num_faces, 3)
    assert vertices.dtype in (np.float32, np.float64)
    assert faces.dtype == np.uint32
    assert np.all(faces < mesh.num_vertices)


def test_mesh_handle_arrays_and_mapping() -> None:
    mesh = load_mesh()

    vertex_handles = mesh.vertex_handles()
    edge_handles = mesh.edge_handles()
    face_handles = mesh.face_handles()

    assert vertex_handles.dtype == np.uint64
    assert edge_handles.dtype == np.uint64
    assert face_handles.dtype == np.uint64
    assert len(vertex_handles) == mesh.num_vertices
    assert len(edge_handles) == mesh.num_edges
    assert len(face_handles) == mesh.num_faces

    vertex = rx.VertexHandle(int(vertex_handles[0]))
    edge = rx.EdgeHandle(int(edge_handles[0]))
    face = rx.FaceHandle(int(face_handles[0]))

    assert 0 <= mesh.global_id(vertex) < mesh.num_vertices
    assert 0 <= mesh.global_id(edge) < mesh.num_edges
    assert 0 <= mesh.global_id(face) < mesh.num_faces
    assert 0 <= mesh.linear_id(vertex) < mesh.num_vertices
    assert 0 <= mesh.linear_id(edge) < mesh.num_edges
    assert 0 <= mesh.linear_id(face) < mesh.num_faces

    assert mesh.owner_handle(vertex).is_valid()
    assert mesh.owner_handle(edge).is_valid()
    assert mesh.owner_handle(face).is_valid()


def test_host_iteration_callbacks() -> None:
    mesh = load_mesh()

    vertices: list[int] = []
    edges: list[int] = []
    faces: list[int] = []

    mesh.for_each_vertex(lambda h: vertices.append(mesh.global_id(h)))
    mesh.for_each_edge(lambda h: edges.append(mesh.global_id(h)))
    mesh.for_each_face(lambda h: faces.append(mesh.global_id(h)))

    assert sorted(vertices) == list(range(mesh.num_vertices))
    assert sorted(edges) == list(range(mesh.num_edges))
    assert sorted(faces) == list(range(mesh.num_faces))


def test_bounding_box_scale_and_save_patcher(tmp_path: Path) -> None:
    mesh = load_mesh()

    lower, upper = mesh.bounding_box()
    assert lower.shape == (3,)
    assert upper.shape == (3,)
    assert np.all(lower <= upper)

    original_lower = lower.copy()
    original_upper = upper.copy()
    mesh.scale([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
    lower, upper = mesh.bounding_box()
    assert lower.shape == (3,)
    assert upper.shape == (3,)
    assert np.all(lower <= upper)
    assert not np.allclose(lower, original_lower) or not np.allclose(upper, original_upper)

    patcher_path = tmp_path / "mesh.patcher"
    mesh.save_patcher(str(patcher_path))
    assert patcher_path.exists()

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))