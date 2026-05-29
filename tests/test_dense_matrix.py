from __future__ import annotations

from pathlib import Path

import numpy as np
import pyrxmesh as rx
import pytest
import torch 

try:
    rx.init()
except RuntimeError as exc:
    if "logger with name 'RXMesh' already exists" not in str(exc):
        raise


def load_mesh() -> rx.RXMeshStatic:
    mesh_path = Path(__file__).resolve().parents[1] / "meshes" / "sphere3.obj"
    assert mesh_path.exists(), f"Missing test mesh: {mesh_path}"
    return rx.RXMeshStatic(str(mesh_path), patch_size=256)


def test_dense_matrix_numpy_round_trip_and_host_view_zero_copy() -> None:
    matrix = rx.DenseMatrix(4, 3, dtype="float32", location=rx.Location.HOST)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)

    matrix.from_numpy(values, target=rx.Location.HOST)
    copied = matrix.to_numpy(source=rx.Location.HOST)
    np.testing.assert_allclose(copied, values)

    view = matrix.to_numpy(source=rx.Location.HOST, copy=False)
    assert view.flags["F_CONTIGUOUS"]
    view[2, 1] = 42.0
    assert matrix.to_numpy(source=rx.Location.HOST)[2, 1] == 42.0


def test_dense_matrix_device_ops_against_numpy() -> None:
    matrix = rx.DenseMatrix(4, 3, dtype="float32", location=rx.Location.ALL)
    other = rx.DenseMatrix(4, 3, dtype="float32", location=rx.Location.ALL)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    other_values = np.full((4, 3), 2.0, dtype=np.float32)

    matrix.from_numpy(values, target=rx.Location.ALL)
    other.from_numpy(other_values, target=rx.Location.ALL)

    np.testing.assert_allclose(matrix.norm2(), np.linalg.norm(values), rtol=1e-5)
    np.testing.assert_allclose(matrix.abs_sum(), np.sum(np.abs(values)), rtol=1e-5)
    np.testing.assert_allclose(matrix.abs_max(), np.max(np.abs(values)), rtol=1e-5)
    np.testing.assert_allclose(matrix.abs_min(), np.min(np.abs(values)), rtol=1e-5)
    np.testing.assert_allclose(matrix.dot(other), np.sum(values * other_values), rtol=1e-5)

    matrix.multiply(2.0)
    matrix.move(rx.Location.DEVICE, rx.Location.HOST)
    np.testing.assert_allclose(
        matrix.to_numpy(source=rx.Location.HOST), values * 2.0, rtol=1e-5
    )

    matrix.axpy(other, 3.0)
    matrix.move(rx.Location.DEVICE, rx.Location.HOST)
    np.testing.assert_allclose(
        matrix.to_numpy(source=rx.Location.HOST),
        values * 2.0 + other_values * 3.0,
        rtol=1e-5,
    )


def test_dense_matrix_torch_dlpack_cpu_zero_copy() -> None:
    matrix = rx.DenseMatrix(4, 3, dtype="float32", location=rx.Location.HOST)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    matrix.from_numpy(values, target=rx.Location.HOST)

    tensor = matrix.to_torch(rx.Location.HOST)
    assert tuple(tensor.shape) == (4, 3)
    assert not tensor.is_cuda
    tensor[1, 2] = 99.0

    assert matrix.to_numpy(source=rx.Location.HOST)[1, 2] == 99.0


def test_dense_matrix_from_torch_copy_cpu_keeps_rxmesh_ownership() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    copied = rx.DenseMatrix.from_torch_copy(values)
    assert copied.shape == (4, 3)
    assert copied.dtype == "float32"
    np.testing.assert_allclose(copied.to_numpy(source=rx.Location.HOST), values.numpy())

    values[:, :] = -1.0
    np.testing.assert_allclose(
        copied.to_numpy(source=rx.Location.HOST),
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )


def test_dense_matrix_from_dlpack_copy_cpu_keeps_rxmesh_ownership() -> None:
    source = rx.DenseMatrix(4, 3, dtype="float32", location=rx.Location.HOST)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    source.from_numpy(values, target=rx.Location.HOST)

    copied = rx.DenseMatrix.from_dlpack_copy(source.to_dlpack(rx.Location.HOST))
    assert copied.shape == source.shape
    assert copied.dtype == source.dtype
    np.testing.assert_allclose(copied.to_numpy(source=rx.Location.HOST), values)

    source_view = source.to_numpy(source=rx.Location.HOST, copy=False)
    source_view[:, :] = -1.0
    np.testing.assert_allclose(copied.to_numpy(source=rx.Location.HOST), values)


def test_dense_matrix_torch_dlpack_cuda_zero_copy() -> None:
    matrix = rx.DenseMatrix(4, 3, dtype="float32", location=rx.Location.ALL)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    matrix.from_numpy(values, target=rx.Location.ALL)

    tensor = matrix.to_torch()
    assert tuple(tensor.shape) == (4, 3)
    assert tensor.is_cuda
    tensor[3, 1] = 123.0
    torch.cuda.synchronize()

    matrix.move(rx.Location.DEVICE, rx.Location.HOST)
    assert matrix.to_numpy(source=rx.Location.HOST)[3, 1] == 123.0


def test_dense_matrix_from_dlpack_copy_cuda_keeps_rxmesh_ownership() -> None:
    source = rx.DenseMatrix(4, 3, dtype="float32", location=rx.Location.ALL)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    source.from_numpy(values, target=rx.Location.ALL)

    copied = rx.DenseMatrix.from_dlpack_copy(source.to_dlpack(rx.Location.DEVICE))
    assert copied.is_device_allocated
    copied.move(rx.Location.DEVICE, rx.Location.HOST)
    np.testing.assert_allclose(copied.to_numpy(source=rx.Location.HOST), values)

    source.reset(-1.0, location=rx.Location.ALL)
    np.testing.assert_allclose(copied.to_numpy(source=rx.Location.HOST), values)


def test_dense_matrix_mesh_constructor_supports_handle_access() -> None:
    mesh = load_mesh()
    matrix = rx.DenseMatrix(
        mesh,
        mesh.num_vertices,
        2,
        dtype="float32",
        location=rx.Location.HOST,
    )

    vertex = rx.VertexHandle(int(mesh.vertex_handles()[0]))
    row = mesh.linear_id(vertex)

    matrix.set_value(vertex, 1, 7.5)
    assert matrix.value(vertex, 1) == pytest.approx(7.5)
    assert matrix.to_numpy(rx.Location.HOST)[row, 1] == pytest.approx(7.5)

    matrix.set_value(row, 0, 3.25)
    assert matrix.value(row, 0) == pytest.approx(3.25)


def test_attribute_dense_matrix_round_trip() -> None:
    mesh = load_mesh()
    attr = mesh.add_vertex_attribute(
        "matrix_bridge_attr",
        dtype="float32",
        dim=2,
        location=rx.Location.ALL,
    )

    values = np.arange(mesh.num_vertices * 2, dtype=np.float32).reshape(-1, 2)
    attr.from_numpy_copy(values, target=rx.Location.ALL)

    matrix = attr.to_matrix_copy()
    assert matrix.shape == attr.shape
    assert matrix.dtype == attr.dtype
    np.testing.assert_allclose(matrix.to_numpy(source=rx.Location.HOST), values)

    updated = values + 7.0
    matrix.from_numpy(updated, target=rx.Location.ALL)
    attr.from_matrix_copy(matrix)
    np.testing.assert_allclose(attr.to_numpy_copy(source=rx.Location.HOST), updated)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
