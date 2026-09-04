import numpy as np
import pyrxmesh as rx
import pytest
import torch


def test_attributes_default_to_soa(mesh) -> None:
    attributes = (
        mesh.add_vertex_attribute("default_soa_vertex"),
        mesh.add_edge_attribute("default_soa_edge"),
        mesh.add_face_attribute("default_soa_face"),
    )

    assert all(attribute.layout == "soa" for attribute in attributes)


def test_attribute_metadata_allocation_and_numpy_round_trip(mesh) -> None:
    attr = mesh.add_vertex_attribute(
        "py_vertex_vec3",
        dtype="float32",
        dim=3,
        location="host",
        layout="soa",
    )

    assert attr.name == "py_vertex_vec3"
    assert attr.dtype == "float32"
    assert attr.element_kind == "vertex"
    assert attr.dim == 3
    assert attr.size == mesh.num_vertices
    assert attr.element_count == mesh.num_vertices
    assert attr.shape == (mesh.num_vertices, 3)
    assert attr.allocated == "host"
    assert attr.layout == "soa"
    assert attr.is_host_allocated
    assert not attr.is_device_allocated
    assert attr.bytes > 0
    assert not hasattr(attr, "from_numpy")
    assert not hasattr(attr, "to_matrix")
    assert not hasattr(attr, "from_matrix")

    assert isinstance(attr, rx.Attribute)
    assert type(attr).__name__ == "VertexAttributeFloat32"

    values = np.arange(mesh.num_vertices * 3, dtype=np.float32).reshape(-1, 3)
    attr.from_numpy_copy(values, target="all")
    assert attr.is_device_allocated

    copied = attr.to_numpy_copy(source="host")
    np.testing.assert_allclose(copied, values)

    copied[:, :] = -1.0
    np.testing.assert_allclose(
        attr.to_numpy_copy(
            source="host"),
        values)

    attr.reset(5.0, location="all")
    np.testing.assert_allclose(attr.to_numpy_copy(source="host"), 5.0)


@pytest.mark.parametrize(
    "layout, name",
    (
        ("soa", "soa"),
        ("aos", "aos"),
        ("aosoa", "aosoa"),
    ),
)
def test_attribute_copy_rows_are_linear_for_every_layout(
    mesh, layout: str, name: str
) -> None:
    attr = mesh.add_vertex_attribute(
        f"linear_order_{name}",
        dtype="float32",
        dim=3,
        location="all",
        layout=layout,
    )
    rows = np.arange(mesh.num_vertices, dtype=np.float32)
    linear_values = np.column_stack(
        (rows, -2.0 * rows - 0.25, 1000.0 + 3.0 * rows)
    ).astype(np.float32)

    attr.from_numpy_copy(linear_values, target="all")

    np.testing.assert_array_equal(
        attr.to_numpy_copy(source="host"), linear_values
    )

    # Keep deliberately different HOST and DEVICE values. The DEVICE copy
    # must gather from device storage without silently reading or overwriting
    # the stale host mirror, for every RXMesh attribute layout.
    attr.reset(-17.0, location="host")
    np.testing.assert_array_equal(
        attr.to_numpy_copy(source="device"), linear_values
    )
    np.testing.assert_array_equal(
        attr.to_numpy_copy(source="host"),
        np.full_like(linear_values, -17.0),
    )

    maximum_handle, maximum = attr.argmax(column=0)
    assert mesh.linear_id(maximum_handle) == mesh.num_vertices - 1
    assert maximum == linear_values[-1, 0]


def test_attribute_rejects_unknown_location_and_layout_strings(mesh) -> None:
    with pytest.raises(ValueError):
        mesh.add_vertex_attribute(
            "invalid_location",
            dtype="float32",
            dim=1,
            location="somewhere",
        )

    with pytest.raises(ValueError):
        mesh.add_vertex_attribute(
            "invalid_layout",
            dtype="float32",
            dim=1,
            layout="interleaved",
        )


def test_attribute_numpy_view_zero_copy(mesh) -> None:
    attr = mesh.add_vertex_attribute(
        "numpy_view_attr",
        dtype="float32",
        dim=3,
        location="host",
        layout="soa",
    )

    values = np.arange(mesh.num_vertices * 3, dtype=np.float32).reshape(-1, 3)
    attr.from_numpy_copy(values, target="host")

    view = attr.to_numpy("host")
    assert view.shape == (mesh.num_vertices, 3)
    assert view.strides == (
        np.dtype(
            np.float32).itemsize,
        mesh.num_vertices *
        np.dtype(
            np.float32).itemsize)
    view[4, 2] = 42.0
    assert attr.to_numpy_copy(source="host")[4, 2] == 42.0


def test_attribute_numpy_view_rejects_aosoa_and_missing_host(mesh) -> None:
    aosoa = mesh.add_vertex_attribute(
        "numpy_view_bad_layout",
        dtype="float32",
        dim=2,
        location="host",
        layout="aosoa",
    )
    with pytest.raises(RuntimeError, match="soa"):
        aosoa.to_numpy("host")

    device_only = mesh.add_vertex_attribute(
        "numpy_view_missing_host",
        dtype="float32",
        dim=2,
        location="device",
        layout="soa",
    )
    with pytest.raises(RuntimeError, match="HOST allocation"):
        device_only.to_numpy("host")

    with pytest.raises(ValueError):
        device_only.to_numpy("device")


def test_attribute_torch_storage_view_zero_copy_host(mesh) -> None:
    attr = mesh.add_vertex_attribute(
        "torch_storage_attr",
        dtype="float32",
        dim=3,
        location="all",
        layout="soa",
    )

    values = np.arange(mesh.num_vertices * 3, dtype=np.float32).reshape(-1, 3)
    attr.from_numpy_copy(values, target="all")

    tensor = attr.to_torch("host")
    assert tuple(tensor.shape) == (mesh.num_vertices, 3)
    assert tensor.dtype == torch.float32
    assert tuple(tensor.stride()) == (1, mesh.num_vertices)
    tensor[:, :] = 13.0

    np.testing.assert_allclose(
        attr.to_numpy_copy(
            source="host"), 13.0)


def test_attribute_torch_storage_view_zero_copy_cuda(mesh) -> None:
    attr = mesh.add_vertex_attribute(
        "torch_storage_attr_cuda",
        dtype="float32",
        dim=2,
        location="all",
        layout="soa",
    )

    values = np.arange(mesh.num_vertices * 2, dtype=np.float32).reshape(-1, 2)
    attr.from_numpy_copy(values, target="all")

    tensor = attr.to_torch("device")
    assert tuple(tensor.shape) == (mesh.num_vertices, 2)
    assert tensor.is_cuda
    tensor[3, 1] = 77.0
    torch.cuda.synchronize()

    attr.move("device", "host")
    assert attr.to_numpy_copy(source="host")[3, 1] == 77.0


def test_attribute_from_torch_copy_cpu(mesh) -> None:
    attr = mesh.add_vertex_attribute(
        "torch_copy_attr",
        dtype="float32",
        dim=2,
        location="all",
    )

    values = torch.arange(mesh.num_vertices * 2,
                          dtype=torch.float32).reshape(-1, 2)
    attr.from_torch_copy(values, target="all")

    np.testing.assert_allclose(
        attr.to_numpy_copy(source="host"),
        values.numpy(),
    )
    expected = values.numpy().copy()
    values[:, :] = -1.0
    np.testing.assert_allclose(
        attr.to_numpy_copy(source="host"),
        expected,
    )


def test_attribute_from_torch_copy_cuda(mesh) -> None:
    attr = mesh.add_vertex_attribute(
        "torch_copy_attr_cuda",
        dtype="float32",
        dim=2,
        location="all",
        layout="soa",
    )

    values = torch.arange(
        mesh.num_vertices * 2, dtype=torch.float32, device="cuda"
    ).reshape(-1, 2)
    attr.from_torch_copy(values, target="all")

    np.testing.assert_allclose(
        attr.to_numpy_copy(source="host"),
        values.cpu().numpy(),
    )


def test_attribute_like_copy_and_remove(mesh) -> None:
    src = mesh.add_edge_attribute(
        "py_edge_src",
        dtype="int32",
        dim=1,
        location="all",
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
    src.from_numpy_copy(values, target="all")
    dst.copy_from(src)
    np.testing.assert_array_equal(
        dst.to_numpy_copy(
            source="host"), values)

    mesh.remove_attribute("py_edge_dst")
    assert not mesh.has_attribute("py_edge_dst")
    assert "py_edge_dst" not in mesh.attribute_names()


def test_attribute_reductions_float32(mesh) -> None:
    attr = mesh.add_vertex_attribute(
        "py_reduce_values",
        dtype="float32",
        dim=1,
        location="all",
    )
    other = mesh.add_attribute_like("py_reduce_other", attr)

    values = np.arange(mesh.num_vertices, dtype=np.float32).reshape(-1, 1)
    other_values = np.full((mesh.num_vertices, 1), 2.0, dtype=np.float32)
    attr.from_numpy_copy(values, target="all")
    other.from_numpy_copy(other_values, target="all")

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
