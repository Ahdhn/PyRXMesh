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


def test_attribute_metadata_allocation_and_numpy_round_trip() -> None:
    mesh = load_mesh()
    attr = mesh.add_vertex_attribute(
        "py_vertex_vec3",
        dtype="float32",
        dim=3,
        location=rx.Location.HOST,
        layout=rx.Layout.SoA,
    )

    assert attr.name == "py_vertex_vec3"
    assert attr.dtype == "float32"
    assert attr.element_kind == "vertex"
    assert attr.dim == 3
    assert attr.size == mesh.num_vertices
    assert attr.element_count == mesh.num_vertices
    assert attr.shape == (mesh.num_vertices, 3)
    assert attr.is_host_allocated
    assert not attr.is_device_allocated
    assert attr.bytes > 0
    assert not hasattr(attr, "from_numpy")
    assert not hasattr(attr, "to_matrix")
    assert not hasattr(attr, "from_matrix")
    
    assert isinstance(attr, rx.Attribute)
    assert type(attr).__name__ == "VertexAttributeFloat32"

    values = np.arange(mesh.num_vertices * 3, dtype=np.float32).reshape(-1, 3)
    attr.from_numpy_copy(values, target=rx.Location.ALL)
    assert attr.is_device_allocated

    copied = attr.to_numpy_copy(source=rx.Location.HOST)
    np.testing.assert_allclose(copied, values)

    copied[:, :] = -1.0
    np.testing.assert_allclose(attr.to_numpy_copy(source=rx.Location.HOST), values)

    attr.reset(5.0, location=rx.Location.ALL)
    np.testing.assert_allclose(attr.to_numpy_copy(source=rx.Location.HOST), 5.0)


def test_attribute_numpy_view_zero_copy() -> None:
    mesh = load_mesh()
    attr = mesh.add_vertex_attribute(
        "numpy_view_attr",
        dtype="float32",
        dim=3,
        location=rx.Location.HOST,
        layout=rx.Layout.SoA,
    )

    values = np.arange(mesh.num_vertices * 3, dtype=np.float32).reshape(-1, 3)
    attr.from_numpy_copy(values, target=rx.Location.HOST)

    view = attr.to_numpy(rx.Location.HOST)
    assert view.shape == (mesh.num_vertices, 3)
    assert view.strides == (np.dtype(np.float32).itemsize, mesh.num_vertices * np.dtype(np.float32).itemsize)
    view[4, 2] = 42.0
    assert attr.to_numpy_copy(source=rx.Location.HOST)[4, 2] == 42.0


def test_attribute_numpy_view_rejects_aosoa_and_missing_host() -> None:
    mesh = load_mesh()
    aosoa = mesh.add_vertex_attribute(
        "numpy_view_bad_layout",
        dtype="float32",
        dim=2,
        location=rx.Location.HOST,
        layout=rx.Layout.AoSoA,
    )
    with pytest.raises(RuntimeError, match="Layout.SoA"):
        aosoa.to_numpy(rx.Location.HOST)

    device_only = mesh.add_vertex_attribute(
        "numpy_view_missing_host",
        dtype="float32",
        dim=2,
        location=rx.Location.DEVICE,
        layout=rx.Layout.SoA,
    )
    with pytest.raises(RuntimeError, match="HOST allocation"):
        device_only.to_numpy(rx.Location.HOST)

    with pytest.raises(ValueError):
        device_only.to_numpy(rx.Location.DEVICE)


def test_attribute_torch_storage_view_zero_copy_host() -> None:
    mesh = load_mesh()
    attr = mesh.add_vertex_attribute(
        "torch_storage_attr",
        dtype="float32",
        dim=3,
        location=rx.Location.ALL,
        layout=rx.Layout.SoA,
    )

    values = np.arange(mesh.num_vertices * 3, dtype=np.float32).reshape(-1, 3)
    attr.from_numpy_copy(values, target=rx.Location.ALL)

    tensor = attr.to_torch(rx.Location.HOST)
    assert tuple(tensor.shape) == (mesh.num_vertices, 3)
    assert tensor.dtype == torch.float32
    assert tuple(tensor.stride()) == (1, mesh.num_vertices)
    tensor[:, :] = 13.0

    np.testing.assert_allclose(attr.to_numpy_copy(source=rx.Location.HOST), 13.0)


def test_attribute_torch_storage_view_zero_copy_cuda() -> None:
    mesh = load_mesh()
    attr = mesh.add_vertex_attribute(
        "torch_storage_attr_cuda",
        dtype="float32",
        dim=2,
        location=rx.Location.ALL,
        layout=rx.Layout.SoA,
    )

    values = np.arange(mesh.num_vertices * 2, dtype=np.float32).reshape(-1, 2)
    attr.from_numpy_copy(values, target=rx.Location.ALL)

    tensor = attr.to_torch(rx.Location.DEVICE)
    assert tuple(tensor.shape) == (mesh.num_vertices, 2)
    assert tensor.is_cuda
    tensor[3, 1] = 77.0
    torch.cuda.synchronize()

    attr.move(rx.Location.DEVICE, rx.Location.HOST)
    assert attr.to_numpy_copy(source=rx.Location.HOST)[3, 1] == 77.0


def test_attribute_from_torch_copy_cpu() -> None:
    mesh = load_mesh()
    attr = mesh.add_vertex_attribute(
        "torch_copy_attr",
        dtype="float32",
        dim=2,
        location=rx.Location.ALL,
    )

    values = torch.arange(mesh.num_vertices * 2, dtype=torch.float32).reshape(-1, 2)
    attr.from_torch_copy(values, target=rx.Location.ALL)

    np.testing.assert_allclose(
        attr.to_numpy_copy(source=rx.Location.HOST),
        values.numpy(),
    )
    expected = values.numpy().copy()
    values[:, :] = -1.0
    np.testing.assert_allclose(
        attr.to_numpy_copy(source=rx.Location.HOST),
        expected,
    )


def test_attribute_from_torch_copy_cuda() -> None:
    mesh = load_mesh()
    attr = mesh.add_vertex_attribute(
        "torch_copy_attr_cuda",
        dtype="float32",
        dim=2,
        location=rx.Location.ALL,
        layout=rx.Layout.SoA,
    )

    values = torch.arange(
        mesh.num_vertices * 2, dtype=torch.float32, device="cuda"
    ).reshape(-1, 2)
    attr.from_torch_copy(values, target=rx.Location.ALL)

    np.testing.assert_allclose(
        attr.to_numpy_copy(source=rx.Location.HOST),
        values.cpu().numpy(),
    )


def test_attribute_like_copy_and_remove() -> None:
    mesh = load_mesh()
    src = mesh.add_edge_attribute(
        "py_edge_src",
        dtype="int32",
        dim=1,
        location=rx.Location.ALL,
    )
    dst = mesh.add_attribute_like("py_edge_dst", src)

    assert mesh.has_attribute("py_edge_src")
    assert mesh.has_attribute("py_edge_dst")
    assert "py_edge_src" in mesh.attribute_names()
    assert "py_edge_dst" in mesh.attribute_names()
    assert dst.dtype == src.dtype
    assert dst.element_kind == src.element_kind
    assert dst.dim == src.dim
    assert dst.shape == src.shape

    values = np.arange(mesh.num_edges, dtype=np.int32).reshape(-1, 1)
    src.from_numpy_copy(values, target=rx.Location.ALL)
    dst.copy_from(src)
    np.testing.assert_array_equal(dst.to_numpy_copy(source=rx.Location.HOST), values)

    mesh.remove_attribute("py_edge_dst")
    assert not mesh.has_attribute("py_edge_dst")
    assert "py_edge_dst" not in mesh.attribute_names()


def test_attribute_reductions_float32() -> None:
    mesh = load_mesh()
    attr = mesh.add_vertex_attribute(
        "py_reduce_values",
        dtype="float32",
        dim=1,
        location=rx.Location.ALL,
    )
    other = mesh.add_attribute_like("py_reduce_other", attr)

    values = np.arange(mesh.num_vertices, dtype=np.float32).reshape(-1, 1)
    other_values = np.full((mesh.num_vertices, 1), 2.0, dtype=np.float32)
    attr.from_numpy_copy(values, target=rx.Location.ALL)
    other.from_numpy_copy(other_values, target=rx.Location.ALL)

    np.testing.assert_allclose(attr.reduce_sum(), np.sum(values), rtol=1e-5)
    np.testing.assert_allclose(attr.reduce_min(), np.min(values), rtol=1e-5)
    np.testing.assert_allclose(attr.reduce_max(), np.max(values), rtol=1e-5)
    np.testing.assert_allclose(attr.norm2(), np.linalg.norm(values), rtol=1e-5)
    np.testing.assert_allclose(
        attr.dot(other), np.sum(values * other_values), rtol=1e-5
    )

    handle, value = attr.argmax(column=0)
    assert handle.is_valid()
    assert 0 <= mesh.linear_id(handle) < mesh.num_vertices
    assert value == np.max(values)

    handle, value = attr.argmin(column=0)
    assert handle.is_valid()
    assert 0 <= mesh.linear_id(handle) < mesh.num_vertices
    assert value == np.min(values)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
