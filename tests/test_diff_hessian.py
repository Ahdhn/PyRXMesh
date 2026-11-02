from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

import pyrxmesh as rx
import rxmesh_diff_energy


def _reference_hessian(mesh: rx.RXMeshStatic, order: str, weight: float):
    faces = np.asarray(mesh.faces(order=order), dtype=np.int64)
    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)])
    )
    edges.sort(axis=1)
    u, v = np.unique(edges, axis=0).T
    rows = np.concatenate((u, v, u, v))
    cols = np.concatenate((u, v, v, u))
    data = np.concatenate((np.ones(2 * len(u)), -np.ones(2 * len(u))))
    laplacian = sp.coo_matrix(
        (data, (rows, cols)), shape=(mesh.num_vertices, mesh.num_vertices)
    ).tocsr()
    return 2 * weight * sp.kron(laplacian, sp.eye(3), format="csr")


def _assert_sparse_close(actual, expected, *, atol=2e-6):
    difference = (actual - expected).tocsr()
    assert not difference.nnz or np.max(np.abs(difference.data)) < atol


def _unavailable(callback, error=RuntimeError):
    try:
        callback()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def test_hessian_evaluation(mesh: rx.RXMeshStatic) -> None:
    n = mesh.num_vertices
    p = np.asarray(mesh.linear_to_global(rx.ElementKind.Vertex), dtype=np.int64)
    assert mesh.num_patches > 1 and np.any(p != np.arange(n))
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    direction = rng.normal(size=(n, 3)).astype(np.float32)
    attr = mesh.add_vertex_attribute(
        "hessian_check_positions", dim=3, location="all", layout="soa"
    )
    attr.from_numpy_copy(x, target="all")
    params = {"w0": 0.37, "w1": 0.19}
    energy = rxmesh_diff_energy.make_energy(mesh, params, with_hessian=True)
    reference = _reference_hessian(mesh, "linear", 0.56)
    expected_gradient = (reference @ x.reshape(-1)).reshape(n, 3)
    expected_loss = 0.5 * x.reshape(-1) @ expected_gradient.reshape(-1)
    first_order = rxmesh_diff_energy.make_energy(mesh, params)
    assert not first_order.hessian_enabled
    assert np.isclose(first_order.evaluate(attr, derivatives=0),
                      expected_loss, rtol=2e-5)
    assert np.isclose(first_order.evaluate(attr), expected_loss, rtol=2e-5)
    _unavailable(lambda: first_order.evaluate(attr, derivatives=2), ValueError)
    _unavailable(lambda: energy.evaluate(attr, derivatives=3), ValueError)
    _unavailable(lambda: rxmesh_diff_energy.make_energy(
        mesh, params, projected_hessian=True), ValueError)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    token = int(stream.cuda_stream)

    def gradient():
        return energy.gradient_snapshot(stream=token).to_torch().cpu().numpy()

    _unavailable(energy.hessian_snapshot)
    _unavailable(lambda: energy.hessian_view)
    loss = energy.evaluate(attr, stream=token, derivatives=0)
    assert np.isclose(loss, expected_loss, rtol=2e-5)
    assert energy.has_evaluated and not energy.has_gradient
    assert not energy.has_hessian
    _unavailable(lambda: energy.gradient_view)
    _unavailable(energy.gradient_snapshot)

    energy.evaluate(attr, stream=token)  # Default remains first order.
    np.testing.assert_allclose(
        gradient(),
        expected_gradient,
        rtol=2e-5,
        atol=2e-5)
    assert energy.has_gradient and not energy.has_hessian
    energy.evaluate(attr, stream=token, derivatives=2)
    hessian = energy.hessian_snapshot(stream=token)
    assert hessian.shape == (3 * n, 3 * n) and hessian.variable_dim == 3
    assert energy.has_gradient and energy.has_hessian
    _assert_sparse_close(hessian.to_scipy_csr(), reference)
    _assert_sparse_close(hessian.to_scipy_csr(), hessian.to_scipy_csr().T)
    np.testing.assert_allclose(
        gradient(),
        expected_gradient,
        rtol=2e-5,
        atol=2e-5)

    # Device CSR exports alias the energy, including indices and values.
    view = energy.hessian_view
    h_torch = view.to_torch("device")
    other = energy.hessian_view.to_torch("device")
    for component in ("crow_indices", "col_indices", "values"):
        assert getattr(h_torch, component)().data_ptr() == \
            getattr(other, component)().data_ptr()
    del other
    assert view.variable_dim == 3
    torch.testing.assert_close(h_torch.values(),
                               hessian.to_torch("device").values())
    # Logical vertex-major DOFs are unchanged, despite SoA input storage.
    vector = torch.as_tensor(direction.reshape(-1, 1), device="cuda")
    hv = torch.sparse.mm(h_torch, vector).cpu().numpy().reshape(-1)
    np.testing.assert_allclose(hv, reference @ direction.reshape(-1),
                               rtol=2e-5, atol=2e-5)
    eps = 1e-2
    attr.from_numpy_copy(x + eps * direction, target="all")
    energy.evaluate(attr, derivatives=1)
    plus = gradient().copy()
    attr.from_numpy_copy(x - eps * direction, target="all")
    energy.evaluate(attr, derivatives=1)
    hv_fd = ((plus - gradient()) / (2 * eps)).reshape(-1)
    relative_error = np.linalg.norm(hv_fd - hv) / np.linalg.norm(hv)
    assert relative_error < 1e-4, relative_error
    _unavailable(energy.hessian_snapshot)
    _unavailable(lambda: energy.hessian_view)

    # Global-order conversion is entirely caller-side and permutes both axes.
    g2l = np.asarray(
        mesh.global_to_linear(
            rx.ElementKind.Vertex),
        dtype=np.int64)
    q = (3 * g2l[:, None] + np.arange(3)).reshape(-1)
    _assert_sparse_close(hessian.to_scipy_csr()[q][:, q],
                         _reference_hessian(mesh, "global", 0.56))

    energy.evaluate(attr, stream=token, derivatives=2)
    hessian.reset(0)
    # Mutations are allowed and immediately visible through a fresh view.
    h_torch.values().zero_()
    assert torch.count_nonzero(
        energy.hessian_view.to_torch("device").values()).item() == 0
    # Releasing/dropping a wrapper must not release the energy's storage.
    view.release()
    del view
    fresh = energy.evaluate_hessian(attr, stream=token)
    assert np.count_nonzero(hessian.values_to_numpy_copy()) == 0
    torch.testing.assert_close(h_torch.values(),
                               fresh.to_torch("device").values())
    del energy
    # Only the exported tensor retains this energy now.
    torch.testing.assert_close(h_torch.values(),
                               fresh.to_torch("device").values())
    _assert_sparse_close(fresh.to_scipy_csr(), reference)

    # Negative quadratics make projection observable: every local eigenvalue
    # is <= 0, so RXMesh clamps it to float32 epsilon=1e-6 before assembly.
    projected = rxmesh_diff_energy.make_energy(
        mesh, {"w0": -0.37, "w1": -0.19},
        with_hessian=True, projected_hessian=True,
    )
    assert projected.hessian_enabled and projected.projected_hessian
    losses = [projected.evaluate(attr, derivatives=d) for d in (0, 1, 2)]
    np.testing.assert_allclose(losses, losses[0], rtol=2e-5)
    projected_h = projected.hessian_snapshot().to_scipy_csr()
    degree = reference.diagonal() / (2 * 0.56)
    _assert_sparse_close(projected_h, sp.diags(2e-6 * degree), atol=1e-9)
    expected = (-reference @ (x - eps * direction).reshape(-1)).reshape(n, 3)
    np.testing.assert_allclose(projected.gradient_snapshot().to_torch().cpu(),
                               expected, rtol=2e-5, atol=2e-5)
    projected.evaluate(attr, derivatives=0)
    _unavailable(projected.hessian_snapshot)
    _unavailable(projected.gradient_snapshot)

    # Hessian-capable specializations still support the original Torch paths.
    xt = torch.tensor(x - eps * direction, device="cuda", requires_grad=True)
    torch_loss = projected.torch(xt)
    torch_loss.backward()
    np.testing.assert_allclose(xt.grad.cpu(), expected, rtol=2e-5, atol=2e-5)
    with torch.no_grad():
        passive_loss = projected.torch(xt)
    torch.testing.assert_close(passive_loss, torch_loss.detach())
    out = torch.empty_like(xt)
    direct_loss = projected.value_and_grad(xt, out=out)
    torch.testing.assert_close(direct_loss, passive_loss)
    np.testing.assert_allclose(out.cpu(), expected, rtol=2e-5, atol=2e-5)
