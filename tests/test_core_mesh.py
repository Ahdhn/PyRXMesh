from pathlib import Path
import weakref

import numpy as np
import pyrxmesh as rx
import pytest


_ORDERING_CASES = (
    (rx.ElementKind.Vertex, "vertex_handles", "num_vertices", rx.VertexHandle),
    (rx.ElementKind.Edge, "edge_handles", "num_edges", rx.EdgeHandle),
    (rx.ElementKind.Face, "face_handles", "num_faces", rx.FaceHandle),
)


def _read_obj_vertices_and_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            face: list[int] = []
            for field in fields[1:]:
                index = int(field.split("/", 1)[0])
                face.append(index - 1 if index > 0 else len(vertices) + index)
            if len(face) != 3:
                raise AssertionError("The ordering fixture must be triangular.")
            faces.append(face)

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.uint32),
    )


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


def test_core_mesh_metadata_and_topology_arrays(mesh) -> None:
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


def test_mesh_handle_arrays_and_mapping(mesh) -> None:
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


@pytest.mark.parametrize(
    "kind, handles_name, count_name, handle_type", _ORDERING_CASES
)
def test_linear_to_global_is_bijection(
    mesh,
    kind,
    handles_name: str,
    count_name: str,
    handle_type,
) -> None:
    count = getattr(mesh, count_name)
    linear_to_global = mesh.linear_to_global(kind)

    assert linear_to_global.shape == (count,)
    assert linear_to_global.dtype == np.uint32
    assert linear_to_global.flags.owndata
    np.testing.assert_array_equal(
        np.sort(linear_to_global), np.arange(count, dtype=np.uint32)
    )

    handles = getattr(mesh, handles_name)()
    expected = np.asarray(
        [mesh.global_id(handle_type(int(value))) for value in handles],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(linear_to_global, expected)


@pytest.mark.parametrize(
    "kind, handles_name, count_name, handle_type", _ORDERING_CASES
)
def test_global_to_linear_is_inverse(
    mesh,
    kind,
    handles_name: str,
    count_name: str,
    handle_type,
) -> None:
    del handles_name, handle_type
    count = getattr(mesh, count_name)
    linear_to_global = mesh.linear_to_global(kind)
    global_to_linear = mesh.global_to_linear(kind)
    linear = np.arange(count, dtype=np.uint32)

    assert global_to_linear.shape == (count,)
    assert global_to_linear.dtype == np.uint32
    assert global_to_linear.flags.owndata
    np.testing.assert_array_equal(global_to_linear[linear_to_global], linear)
    np.testing.assert_array_equal(linear_to_global[global_to_linear], linear)


def test_ordering_fixture_has_nonidentity_multi_patch_permutation(mesh) -> None:
    assert mesh.num_patches > 1
    permutations = (
        mesh.linear_to_global(rx.ElementKind.Vertex),
        mesh.linear_to_global(rx.ElementKind.Edge),
        mesh.linear_to_global(rx.ElementKind.Face),
    )
    assert any(
        not np.array_equal(mapping, np.arange(len(mapping), dtype=np.uint32))
        for mapping in permutations
    )


def test_mesh_vertices_are_in_declared_linear_order(
        mesh, mesh_path: Path) -> None:
    input_vertices, _ = _read_obj_vertices_and_faces(mesh_path)
    linear_to_global = mesh.linear_to_global(rx.ElementKind.Vertex)
    actual = mesh.vertices()

    np.testing.assert_allclose(
        actual,
        input_vertices[linear_to_global],
        rtol=1e-6,
        atol=1e-7,
    )
    assert actual.flags.owndata
    assert actual.flags.c_contiguous


def test_mesh_vertices_can_be_written_directly_in_global_order(
    mesh, mesh_path: Path
) -> None:
    input_vertices, _ = _read_obj_vertices_and_faces(mesh_path)
    actual = mesh.vertices(order="global")

    np.testing.assert_allclose(actual, input_vertices, rtol=1e-6, atol=1e-7)
    assert actual.flags.owndata
    assert actual.flags.c_contiguous


def test_mesh_faces_rows_and_vertex_ids_are_declared_linear(
    mesh, mesh_path: Path
) -> None:
    _, input_faces = _read_obj_vertices_and_faces(mesh_path)
    linear_to_global_faces = mesh.linear_to_global(rx.ElementKind.Face)
    global_to_linear_vertices = mesh.global_to_linear(rx.ElementKind.Vertex)

    expected = global_to_linear_vertices[
        input_faces[linear_to_global_faces]
    ]
    actual = mesh.faces()
    np.testing.assert_array_equal(
        np.sort(actual, axis=1), np.sort(expected, axis=1)
    )
    assert actual.flags.owndata
    assert actual.flags.c_contiguous


def test_mesh_faces_can_be_written_directly_in_global_order(
    mesh, mesh_path: Path
) -> None:
    _, input_faces = _read_obj_vertices_and_faces(mesh_path)
    actual = mesh.faces(order="global")

    np.testing.assert_array_equal(
        np.sort(actual, axis=1), np.sort(input_faces, axis=1)
    )
    assert actual.flags.owndata
    assert actual.flags.c_contiguous


def test_mesh_array_order_is_validated(mesh) -> None:
    with pytest.raises(ValueError, match="linear.*global"):
        mesh.vertices(order="input")
    with pytest.raises(ValueError, match="linear.*global"):
        mesh.faces(order="input")


def test_mesh_supports_weak_ordering_cache_keys(mesh) -> None:
    reference = weakref.ref(mesh)
    cache = weakref.WeakKeyDictionary()
    cache[mesh] = mesh.linear_to_global(rx.ElementKind.Vertex)

    assert reference() is mesh
    assert cache[mesh].dtype == np.uint32


def test_host_iteration_callbacks(mesh) -> None:
    vertices: list[int] = []
    edges: list[int] = []
    faces: list[int] = []

    mesh.for_each_vertex(lambda h: vertices.append(mesh.global_id(h)))
    mesh.for_each_edge(lambda h: edges.append(mesh.global_id(h)))
    mesh.for_each_face(lambda h: faces.append(mesh.global_id(h)))

    assert sorted(vertices) == list(range(mesh.num_vertices))
    assert sorted(edges) == list(range(mesh.num_edges))
    assert sorted(faces) == list(range(mesh.num_faces))


def test_bounding_box_scale_and_save_patcher(mesh, tmp_path: Path) -> None:
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
    assert not np.allclose(
        lower, original_lower) or not np.allclose(
        upper, original_upper)

    patcher_path = tmp_path / "mesh.patcher"
    mesh.save_patcher(str(patcher_path))
    assert patcher_path.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
