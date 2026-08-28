from __future__ import annotations

from pathlib import Path

import numpy as np
import pyrxmesh as rx
import pytest
import torch


try:
    import rxmesh_diff_energy
except ImportError as exc:
    raise ImportError(
        "Install examples/diff_energy_plugin before running this test module."
    ) from exc


_TETRAHEDRON_OBJ = """\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""


@pytest.fixture()
def tiny_mesh(tmp_path: Path) -> rx.RXMeshStatic:
    path = tmp_path / "diff_autograd_tetrahedron.obj"
    path.write_text(_TETRAHEDRON_OBJ, encoding="utf-8")
    return rx.RXMeshStatic(str(path), patch_size=32)


def _positions(mesh: rx.RXMeshStatic, phase: float = 0.0) -> np.ndarray:
    indices = np.arange(mesh.num_vertices * 3, dtype=np.float64).reshape(-1, 3)
    return (
        0.7 * np.sin(0.37 * indices + phase)
        + 0.2 * np.cos(0.11 * indices - 0.3 * phase)
    ).astype(np.float32)


def _unique_edges(mesh: rx.RXMeshStatic) -> np.ndarray:
    faces = np.asarray(mesh.faces(), dtype=np.int64)
    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _reference(mesh, positions, *, weight=0.56):
    edges = _unique_edges(mesh)
    x = np.asarray(positions, dtype=np.float64)
    delta = x[edges[:, 0]] - x[edges[:, 1]]
    loss = weight * np.einsum("ij,ij->", delta, delta)
    gradient = np.zeros_like(x)
    contribution = 2.0 * weight * delta
    np.add.at(gradient, edges[:, 0], contribution)
    np.add.at(gradient, edges[:, 1], -contribution)
    return float(loss), gradient.astype(np.float32)


def _energy(mesh, weight=0.56):
    return rxmesh_diff_energy.make_energy(
        mesh, {"w0": 0.37 * weight / 0.56, "w1": 0.19 * weight / 0.56}
    )


def _attribute(mesh, name, values, layout=rx.Layout.SoA):
    attr = mesh.add_vertex_attribute(
        name,
        dtype="float32",
        dim=3,
        location=rx.Location.ALL,
        layout=layout,
    )
    attr.from_numpy_copy(values, target=rx.Location.ALL)
    return attr


def _row_variable(values):
    return torch.tensor(
        values, dtype=torch.float32, device="cuda", requires_grad=True
    )


def _soa_variable(values):
    result = rx.diff.empty_soa(values.shape, dtype=torch.float32, device="cuda")
    result.copy_(torch.as_tensor(values, dtype=torch.float32, device="cuda"))
    return result.detach().requires_grad_()


def _assert_gradient(actual, expected, scale=1.0):
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        scale * expected,
        rtol=5e-4,
        atol=1e-5,
    )


def _stream_token(stream):
    raw = int(stream.cuda_stream)
    return 1 if raw == 0 else raw


def test_low_level_results_require_evaluate(tiny_mesh):
    energy = _energy(tiny_mesh)
    assert not energy.has_evaluated
    with pytest.raises(RuntimeError, match="before evaluate"):
        _ = energy.loss
    with pytest.raises(RuntimeError, match="before evaluate"):
        _ = energy.gradient_view
    with pytest.raises(RuntimeError, match="before evaluate"):
        energy.gradient_snapshot()


@pytest.mark.parametrize("layout",
                         [rx.Layout.SoA,
                          rx.Layout.AoS,
                          rx.Layout.AoSoA])
def test_low_level_evaluate_matches_reference(tiny_mesh, layout):
    values = _positions(tiny_mesh, 0.31)
    expected_loss, expected_gradient = _reference(tiny_mesh, values)
    energy = _energy(tiny_mesh)
    attr = _attribute(tiny_mesh, f"positions_{int(layout)}", values, layout)

    assert energy.evaluate(attr) == pytest.approx(expected_loss, rel=5e-4)
    assert energy.has_evaluated
    view = energy.gradient_view
    assert view.is_read_only
    _assert_gradient(view.to_torch(rx.Location.DEVICE), expected_gradient)


def test_gradient_snapshot_is_stable(tiny_mesh):
    first = _positions(tiny_mesh, 0.1)
    second = _positions(tiny_mesh, 1.2)
    energy = _energy(tiny_mesh)
    attr = _attribute(tiny_mesh, "snapshot_positions", first)
    energy.evaluate(attr)
    snapshot = energy.gradient_snapshot().to_torch(rx.Location.DEVICE).clone()

    attr.from_numpy_copy(second, target=rx.Location.ALL)
    energy.evaluate(attr)
    _, expected_first = _reference(tiny_mesh, first)
    _assert_gradient(snapshot, expected_first)


def test_direct_torch_forward_writes_caller_buffers(tiny_mesh):
    values = _positions(tiny_mesh, 0.7)
    expected_loss, expected_gradient = _reference(tiny_mesh, values)
    energy = _energy(tiny_mesh)
    x = _soa_variable(values)
    gradient = torch.empty_like(x, memory_format=torch.contiguous_format)
    term_losses = torch.empty(energy.term_count, dtype=x.dtype, device=x.device)
    stream = torch.cuda.current_stream()

    energy._torch_forward(
        x.data_ptr(),
        gradient.data_ptr(),
        term_losses.data_ptr(),
        _stream_token(stream),
    )
    stream.synchronize()
    assert term_losses.sum().item() == pytest.approx(expected_loss, rel=5e-4)
    _assert_gradient(gradient, expected_gradient)
    assert not energy.has_evaluated


def test_direct_torch_forward_without_gradient(tiny_mesh):
    values = _positions(tiny_mesh, 0.7)
    expected_loss = _reference(tiny_mesh, values)[0]
    energy = _energy(tiny_mesh)
    x = _soa_variable(values)
    term_losses = torch.empty(energy.term_count, dtype=x.dtype, device=x.device)
    stream = torch.cuda.current_stream()

    energy._torch_forward(
        x.data_ptr(), 0, term_losses.data_ptr(), _stream_token(stream)
    )
    stream.synchronize()
    assert term_losses.sum().item() == pytest.approx(expected_loss, rel=5e-4)
    assert not energy.has_evaluated


def test_autograd_stages_row_major_input(tiny_mesh):
    values = _positions(tiny_mesh, 0.4)
    expected_loss, expected_gradient = _reference(tiny_mesh, values)
    x = _row_variable(values)
    loss = _energy(tiny_mesh).torch(x)
    loss.backward()

    assert loss.item() == pytest.approx(expected_loss, rel=5e-4)
    _assert_gradient(x.grad, expected_gradient)


def test_autograd_borrows_strict_soa_input(tiny_mesh):
    values = _positions(tiny_mesh, 0.8)
    expected_loss, expected_gradient = _reference(tiny_mesh, values)
    x = _soa_variable(values)
    loss = _energy(tiny_mesh).torch(x, copy="never")
    loss.backward()

    assert x.stride() == (1, tiny_mesh.num_vertices)
    assert loss.item() == pytest.approx(expected_loss, rel=5e-4)
    _assert_gradient(x.grad, expected_gradient)


def test_strict_copy_policy_rejects_row_major(tiny_mesh):
    x = _row_variable(_positions(tiny_mesh))
    with pytest.raises(ValueError, match="SoA strides"):
        _energy(tiny_mesh).torch(x, copy="never")


def test_strict_soa_accepts_nonzero_storage_offset(tiny_mesh):
    values = _positions(tiny_mesh, 0.9)
    n = tiny_mesh.num_vertices
    storage = torch.empty(n * 3 + 1, dtype=torch.float32, device="cuda")
    x = torch.as_strided(storage, values.shape, (1, n), storage_offset=1)
    x.copy_(torch.as_tensor(values, device="cuda"))
    x = x.detach().requires_grad_()
    gradient = torch.autograd.grad(
        _energy(tiny_mesh).torch(x, copy="never"), x
    )[0]
    _assert_gradient(gradient, _reference(tiny_mesh, values)[1])


def test_input_validation(tiny_mesh):
    energy = _energy(tiny_mesh)
    n = tiny_mesh.num_vertices
    with pytest.raises(ValueError, match="shape"):
        energy.torch(torch.empty((n + 1, 3), device="cuda"))
    with pytest.raises(TypeError, match="dtype"):
        energy.torch(torch.empty((n, 3), dtype=torch.float64, device="cuda"))
    with pytest.raises(ValueError, match="CUDA"):
        energy.torch(torch.empty((n, 3)))
    with pytest.raises(ValueError, match="copy"):
        energy.torch(torch.empty((n, 3), device="cuda"), copy="always")


def test_two_forwards_keep_independent_gradients(tiny_mesh):
    first_values = _positions(tiny_mesh, 0.1)
    second_values = _positions(tiny_mesh, 1.1)
    energy = _energy(tiny_mesh)
    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    with torch.cuda.stream(first_stream):
        first = _row_variable(first_values)
        first_loss = energy.torch(first)
    with torch.cuda.stream(second_stream):
        second = _row_variable(second_values)
        second_loss = energy.torch(second)

    torch.cuda.current_stream().wait_stream(first_stream)
    torch.cuda.current_stream().wait_stream(second_stream)
    second_loss.backward(retain_graph=True)
    first_loss.backward()
    _assert_gradient(first.grad, _reference(tiny_mesh, first_values)[1])
    _assert_gradient(second.grad, _reference(tiny_mesh, second_values)[1])


def test_chain_rule_and_two_energy_sum(tiny_mesh):
    values = _positions(tiny_mesh, 0.6)
    x = _row_variable(values)
    first = _energy(tiny_mesh, 0.4)
    second = _energy(tiny_mesh, 0.7)
    loss = 2.5 * (first.torch(x) + second.torch(x))
    loss.backward()
    expected = _reference(tiny_mesh, values, weight=1.1)[1]
    _assert_gradient(x.grad, expected, scale=2.5)


def test_global_order_conversion_is_differentiable(mesh):
    rng = np.random.default_rng(42)
    global_values = rng.normal(size=(mesh.num_vertices, 3)).astype(np.float32)
    x_global = _row_variable(global_values)
    x_linear = rx.diff.to_linear_soa(mesh, x_global)
    assert x_linear.stride() == (1, mesh.num_vertices)

    loss = _energy(mesh).torch(x_linear, copy="never")
    actual = torch.autograd.grad(loss, x_global)[0]

    linear_to_global = np.asarray(
        mesh.linear_to_global(rx.ElementKind.Vertex), dtype=np.int64
    )
    linear_values = global_values[linear_to_global]
    linear_gradient = _reference(mesh, linear_values)[1]
    expected = np.empty_like(linear_gradient)
    expected[linear_to_global] = linear_gradient
    _assert_gradient(actual, expected)


def test_element_order_index_is_cached_per_mesh_and_element(mesh):
    cache = rx.diff._LINEAR_TO_GLOBAL_INDICES
    cache.pop(mesh, None)
    device = _row_variable(_positions(mesh)).device

    first = rx.diff._linear_to_global_index(mesh, "vertex", device)
    second = rx.diff._linear_to_global_index(mesh, "vertex", device)

    assert first is second
    assert set(cache[mesh]) == {rx.ElementKind.Vertex}


def test_nondefault_forward_and_backward_streams(tiny_mesh):
    values = _positions(tiny_mesh, 0.45)
    x = _row_variable(values)
    energy = _energy(tiny_mesh)
    forward_stream = torch.cuda.Stream()
    backward_stream = torch.cuda.Stream()
    forward_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(forward_stream):
        loss = energy.torch(x)
    with torch.cuda.stream(backward_stream):
        gradient = torch.autograd.grad(loss, x)[0]
    torch.cuda.current_stream().wait_stream(backward_stream)
    _assert_gradient(gradient, _reference(tiny_mesh, values)[1])


def test_double_backward_is_rejected(tiny_mesh):
    x = _row_variable(_positions(tiny_mesh))
    gradient = torch.autograd.grad(
        _energy(tiny_mesh).torch(x), x, create_graph=True
    )[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), x)


def test_empty_energy_returns_zero(tiny_mesh):
    x = _row_variable(_positions(tiny_mesh))
    energy = rxmesh_diff_energy.make_empty_energy(tiny_mesh)
    loss = energy.torch(x)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(x.grad).item() == 0


@pytest.mark.parametrize("context", ["forward_only", "no_grad"])
def test_forward_without_autograd(tiny_mesh, context):
    values = _positions(tiny_mesh, 0.33)
    x = torch.as_tensor(values, dtype=torch.float32, device="cuda")
    energy = _energy(tiny_mesh)
    if context == "no_grad":
        x.requires_grad_()
        with torch.no_grad():
            loss = energy.torch(x)
    else:
        loss = energy.torch(x)
    assert not loss.requires_grad
    assert loss.item() == pytest.approx(
        _reference(tiny_mesh, values)[0], rel=5e-4)
