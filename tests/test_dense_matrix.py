from __future__ import annotations

import numpy as np
import pyrxmesh as rx
import pytest
import torch 


def test_dense_matrix_numpy_round_trip_and_host_view_zero_copy() -> None:
    matrix = rx.DenseMatrix(4, 3, dtype="float32", location="host")
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert matrix.order == "col_major"
    assert matrix.location == "host"

    matrix.from_numpy_copy(values, target="host")
    copied = matrix.to_numpy_copy(source="host")
    np.testing.assert_allclose(copied, values)
    copied[0, 0] = -5.0
    assert matrix.to_numpy("host")[0, 0] == 0.0

    view = matrix.to_numpy("host")
    assert view.flags["F_CONTIGUOUS"]
    view[2, 1] = 42.0
    assert matrix.to_numpy_copy(source="host")[2, 1] == 42.0


def test_dense_matrix_row_major_numpy_round_trip_and_host_view_zero_copy() -> None:
    matrix = rx.DenseMatrix(
        4,
        3,
        dtype="float32",
        location="host",
        order="row_major",
    )
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert matrix.order == "row_major"

    matrix.from_numpy_copy(values, target="host")
    copied = matrix.to_numpy_copy(source="host")
    np.testing.assert_allclose(copied, values)

    view = matrix.to_numpy("host")
    assert view.flags["C_CONTIGUOUS"]
    assert not view.flags["F_CONTIGUOUS"]
    assert view.strides == (values.strides[0], values.strides[1])
    view[2, 1] = 42.0
    assert matrix.to_numpy_copy(source="host")[2, 1] == 42.0


def test_dense_matrix_numpy_view_requires_host_allocation() -> None:
    matrix = rx.DenseMatrix(2, 2, dtype="float32", location="device")
    with pytest.raises(RuntimeError, match="HOST allocation"):
        matrix.to_numpy("host")


def test_dense_matrix_device_ops_against_numpy() -> None:
    stream = 2
    matrix = rx.DenseMatrix(4, 3, dtype="float32", location="all")
    other = rx.DenseMatrix(4, 3, dtype="float32", location="all")
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    other_values = np.full((4, 3), 2.0, dtype=np.float32)

    matrix.from_numpy_copy(values, target="all", stream=stream)
    other.from_numpy_copy(other_values, target="all", stream=stream)
    rx.cuda_stream_synchronize(stream)

    np.testing.assert_allclose(
        matrix.norm2(stream=stream), np.linalg.norm(values), rtol=1e-5
    )
    np.testing.assert_allclose(
        matrix.abs_sum(stream=stream), np.sum(np.abs(values)), rtol=1e-5
    )
    np.testing.assert_allclose(
        matrix.abs_max(stream=stream), np.max(np.abs(values)), rtol=1e-5
    )
    np.testing.assert_allclose(
        matrix.abs_min(stream=stream), np.min(np.abs(values)), rtol=1e-5
    )
    np.testing.assert_allclose(
        matrix.dot(other, stream=stream), np.sum(values * other_values), rtol=1e-5
    )

    matrix.multiply(2.0, stream=stream)
    matrix.move("device", "host", stream=stream)
    rx.cuda_stream_synchronize(stream)
    np.testing.assert_allclose(
        matrix.to_numpy_copy(source="host"), values * 2.0, rtol=1e-5
    )

    matrix.axpy(other, 3.0, stream=stream)
    matrix.move("device", "host", stream=stream)
    rx.cuda_stream_synchronize(stream)
    np.testing.assert_allclose(
        matrix.to_numpy_copy(source="host"),
        values * 2.0 + other_values * 3.0,
        rtol=1e-5,
    )

    copied = rx.DenseMatrix(4, 3, dtype="float32", location="all")
    copied.copy_from(
        matrix,
        source="device",
        target="device",
        stream=stream,
    )
    copied.move("device", "host", stream=stream)
    rx.cuda_stream_synchronize(stream)
    np.testing.assert_allclose(
        copied.to_numpy_copy(source="host"),
        values * 2.0 + other_values * 3.0,
        rtol=1e-5,
    )

    matrix.swap(other, stream=stream)
    matrix.move("device", "host", stream=stream)
    other.move("device", "host", stream=stream)
    rx.cuda_stream_synchronize(stream)
    np.testing.assert_allclose(
        matrix.to_numpy_copy(source="host"), other_values, rtol=1e-5
    )
    np.testing.assert_allclose(
        other.to_numpy_copy(source="host"),
        values * 2.0 + other_values * 3.0,
        rtol=1e-5,
    )

    matrix.reset(4.0, location="all", stream=stream)
    matrix.move("device", "host", stream=stream)
    rx.cuda_stream_synchronize(stream)
    np.testing.assert_allclose(
        matrix.to_numpy_copy(source="host"),
        np.full((4, 3), 4.0, dtype=np.float32),
        rtol=1e-5,
    )

    with pytest.raises(ValueError, match="stream value 0"):
        matrix.multiply(1.0, stream=0)


def test_dense_matrix_torch_dlpack_cpu_zero_copy() -> None:
    matrix = rx.DenseMatrix(4, 3, dtype="float32", location="host")
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    matrix.from_numpy_copy(values, target="host")

    tensor = matrix.to_torch("host")
    assert tuple(tensor.shape) == (4, 3)
    assert not tensor.is_cuda
    tensor[1, 2] = 99.0

    assert matrix.to_numpy_copy(source="host")[1, 2] == 99.0


def test_dense_matrix_row_major_torch_dlpack_cpu_zero_copy() -> None:
    matrix = rx.DenseMatrix(
        4,
        3,
        dtype="float32",
        location="host",
        order="row_major",
    )
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    matrix.from_numpy_copy(values, target="host")

    tensor = matrix.to_torch("host")
    assert tuple(tensor.shape) == (4, 3)
    assert tensor.stride() == (3, 1)
    assert not tensor.is_cuda
    tensor[1, 2] = 99.0

    assert matrix.to_numpy_copy(source="host")[1, 2] == 99.0


def test_dense_matrix_from_torch_copy_cpu_keeps_rxmesh_ownership() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    copied = rx.DenseMatrix.from_torch_copy(values)
    assert copied.shape == (4, 3)
    assert copied.dtype == "float32"
    assert copied.order == "col_major"
    np.testing.assert_allclose(copied.to_numpy_copy(source="host"), values.numpy())

    values[:, :] = -1.0
    np.testing.assert_allclose(
        copied.to_numpy_copy(source="host"),
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )


def test_dense_matrix_from_torch_copy_cpu_supports_row_major() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    copied = rx.DenseMatrix.from_torch_copy(values, order="row_major")
    assert copied.shape == (4, 3)
    assert copied.dtype == "float32"
    assert copied.order == "row_major"
    np.testing.assert_allclose(
        copied.to_numpy_copy(source="host"),
        values.numpy(),
    )

    values[:, :] = -1.0
    np.testing.assert_allclose(
        copied.to_numpy_copy(source="host"),
        np.arange(12, dtype=np.float32).reshape(3, 4).T,
    )


def test_dense_matrix_from_torch_view_cpu_shares_row_major_memory() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    view = rx.DenseMatrix.from_torch_view(values, order="row_major")

    assert view.is_view
    assert view.order == "row_major"
    assert view.shape == (4, 3)
    assert view.is_host_allocated
    assert not view.is_device_allocated
    np.testing.assert_allclose(view.to_numpy_copy("host"), values.numpy())

    values[1, 2] = 77.0
    assert view.value(1, 2) == pytest.approx(77.0)

    view.set_value(2, 1, 88.0)
    assert values[2, 1].item() == pytest.approx(88.0)

    view.from_numpy_copy(
        np.full((4, 3), 5.0, dtype=np.float32),
        target="all",
    )
    assert torch.allclose(values, torch.full_like(values, 5.0))

    with pytest.raises(ValueError, match="external memory view"):
        view.move("host", "device")


def test_dense_matrix_from_torch_view_cpu_supports_col_major_memory() -> None:
    values = torch.empty_strided((4, 3), (1, 4), dtype=torch.float32)
    values.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3))
    view = rx.DenseMatrix.from_torch_view(values, order="col_major")

    assert view.is_view
    assert view.order == "col_major"
    tensor = view.to_torch("host")
    assert tensor.stride() == (1, 4)

    view.set_value(3, 2, 44.0)
    assert values[3, 2].item() == pytest.approx(44.0)


def test_dense_matrix_from_torch_view_rejects_noncompact_strides() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    with pytest.raises(ValueError, match="compact row_major strides"):
        rx.DenseMatrix.from_torch_view(values, order="row_major")


def test_dense_matrix_from_dlpack_copy_cpu_keeps_rxmesh_ownership() -> None:
    source = rx.DenseMatrix(4, 3, dtype="float32", location="host")
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    source.from_numpy_copy(values, target="host")

    copied = rx.DenseMatrix.from_dlpack_copy(source.to_dlpack("host"))
    assert copied.shape == source.shape
    assert copied.dtype == source.dtype
    np.testing.assert_allclose(copied.to_numpy_copy(source="host"), values)

    source_view = source.to_numpy("host")
    source_view[:, :] = -1.0
    np.testing.assert_allclose(copied.to_numpy_copy(source="host"), values)


def test_dense_matrix_torch_dlpack_cuda_zero_copy() -> None:
    matrix = rx.DenseMatrix(4, 3, dtype="float32", location="all")
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    matrix.from_numpy_copy(values, target="all")

    tensor = matrix.to_torch()
    assert tuple(tensor.shape) == (4, 3)
    assert tensor.is_cuda
    tensor[3, 1] = 123.0
    torch.cuda.synchronize()

    matrix.move("device", "host")
    rx.cuda_stream_synchronize()
    assert matrix.to_numpy_copy(source="host")[3, 1] == 123.0


def test_dense_matrix_row_major_torch_dlpack_cuda_zero_copy() -> None:
    matrix = rx.DenseMatrix(
        4,
        3,
        dtype="float32",
        location="all",
        order="row_major",
    )
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    matrix.from_numpy_copy(values, target="all")

    tensor = matrix.to_torch()
    assert tuple(tensor.shape) == (4, 3)
    assert tensor.stride() == (3, 1)
    assert tensor.is_cuda
    tensor[3, 1] = 123.0
    torch.cuda.synchronize()

    matrix.move("device", "host")
    rx.cuda_stream_synchronize()
    assert matrix.to_numpy_copy(source="host")[3, 1] == 123.0


def test_dense_matrix_from_dlpack_copy_cuda_keeps_rxmesh_ownership() -> None:
    source = rx.DenseMatrix(4, 3, dtype="float32", location="all")
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    source.from_numpy_copy(values, target="all")

    copied = rx.DenseMatrix.from_dlpack_copy(source.to_dlpack("device"))
    assert copied.is_device_allocated
    copied.move("device", "host")
    rx.cuda_stream_synchronize()
    np.testing.assert_allclose(copied.to_numpy_copy(source="host"), values)

    source.reset(-1.0, location="all")
    np.testing.assert_allclose(copied.to_numpy_copy(source="host"), values)


def test_dense_matrix_from_torch_copy_cuda_supports_row_major() -> None:
    values = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)
    copied = rx.DenseMatrix.from_torch_copy(values, order="row_major")
    assert copied.is_device_allocated
    assert copied.order == "row_major"
    copied.move("device", "host")
    rx.cuda_stream_synchronize()
    np.testing.assert_allclose(
        copied.to_numpy_copy(source="host"),
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )


def test_dense_matrix_from_torch_view_cuda_shares_row_major_memory() -> None:
    values = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)
    view = rx.DenseMatrix.from_torch_view(values, order="row_major")

    assert view.is_view
    assert view.order == "row_major"
    assert view.shape == (4, 3)
    assert view.is_device_allocated
    assert not view.is_host_allocated

    tensor = view.to_torch()
    assert tensor.is_cuda
    assert tensor.stride() == (3, 1)
    tensor[2, 1] = 77.0
    torch.cuda.synchronize()
    assert values[2, 1].item() == pytest.approx(77.0)

    view.reset(3.0, location="all")
    torch.cuda.synchronize()
    assert torch.allclose(values, torch.full_like(values, 3.0))

    with pytest.raises(ValueError, match="HOST allocation"):
        view.value(0, 0)
    with pytest.raises(ValueError, match="external memory view"):
        view.move("device", "host")


def test_dense_matrix_from_torch_copy_cuda_respects_producer_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = torch.empty((4, 3), dtype=torch.float32, device="cuda")
        values.copy_(torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3))

    copied = rx.DenseMatrix.from_torch_copy(values)
    copied.move("device", "host")
    rx.cuda_stream_synchronize()
    np.testing.assert_allclose(
        copied.to_numpy_copy(source="host"),
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )


def test_dense_matrix_from_dlpack_copy_cuda_passes_stream_to_producer() -> None:
    class Probe:
        def __init__(self, tensor):
            self.tensor = tensor
            self.streams = []

        def __dlpack_device__(self):
            return self.tensor.__dlpack_device__()

        def __dlpack__(self, stream=None):
            self.streams.append(stream)
            return self.tensor.__dlpack__(stream=stream)

    values = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)
    probe = Probe(values)
    copied = rx.DenseMatrix.from_dlpack_copy(probe)

    assert probe.streams
    assert probe.streams[0] is not None
    copied.move("device", "host")
    rx.cuda_stream_synchronize()
    np.testing.assert_allclose(
        copied.to_numpy_copy(source="host"),
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )


def test_dense_matrix_mesh_constructor_supports_handle_access(mesh) -> None:
    matrix = rx.DenseMatrix(
        mesh,
        mesh.num_vertices,
        2,
        dtype="float32",
        location="host",
    )

    vertex = rx.VertexHandle(int(mesh.vertex_handles()[0]))
    row = mesh.linear_id(vertex)

    matrix.set_value(vertex, 1, 7.5)
    assert matrix.value(vertex, 1) == pytest.approx(7.5)
    assert matrix.to_numpy_copy("host")[row, 1] == pytest.approx(7.5)

    matrix.set_value(row, 0, 3.25)
    assert matrix.value(row, 0) == pytest.approx(3.25)


def test_attribute_dense_matrix_round_trip(mesh) -> None:
    attr = mesh.add_vertex_attribute(
        "matrix_bridge_attr",
        dtype="float32",
        dim=2,
        location="all",
    )

    values = np.arange(mesh.num_vertices * 2, dtype=np.float32).reshape(-1, 2)
    attr.from_numpy_copy(values, target="all")

    matrix = attr.to_matrix_copy()
    assert matrix.shape == attr.shape
    assert matrix.dtype == attr.dtype
    np.testing.assert_allclose(matrix.to_numpy_copy(source="host"), values)

    updated = values + 7.0
    matrix.from_numpy_copy(updated, target="all")
    attr.from_matrix_copy(matrix)
    np.testing.assert_allclose(attr.to_numpy_copy(source="host"), updated)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
