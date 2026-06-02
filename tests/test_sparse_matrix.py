from __future__ import annotations

from pathlib import Path

import numpy as np
import pyrxmesh as rx
import pytest
import torch 
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


def test_sparse_matrix_metadata_and_csr_arrays() -> None:
    mesh = load_mesh()
    matrix = mesh.sparse_matrix(rx.Op.VV, dtype="float32")

    assert matrix.rows == mesh.num_vertices
    assert matrix.cols == mesh.num_vertices
    assert matrix.shape == (mesh.num_vertices, mesh.num_vertices)
    assert matrix.nnz > mesh.num_vertices
    assert matrix.dtype == "float32"
    assert matrix.index_dtype == "int32"
    assert matrix.is_host_allocated
    assert matrix.is_device_allocated    

    row_ptr, col_idx, values = matrix.to_numpy_copy()
    assert row_ptr.dtype == np.int32
    assert col_idx.dtype == np.int32
    assert values.dtype == np.float32
    assert row_ptr.shape == (matrix.rows + 1,)
    assert col_idx.shape == (matrix.nnz,)
    assert values.shape == (matrix.nnz,)
    assert row_ptr[0] == 0
    assert row_ptr[-1] == matrix.nnz
    assert np.all(row_ptr[1:] >= row_ptr[:-1])
    assert np.all((col_idx >= 0) & (col_idx < matrix.cols))


def test_sparse_matrix_host_values_round_trip_and_zero_copy_view() -> None:
    mesh = load_mesh()
    matrix = rx.SparseMatrix(mesh, rx.Op.VV, dtype="float64")

    values = np.linspace(0.0, 1.0, matrix.nnz, dtype=np.float64)
    matrix.from_numpy_values_copy(values, target=rx.Location.ALL)
    np.testing.assert_allclose(matrix.to_numpy_copy()[2], values)

    view = matrix.to_numpy()[2]
    view[0] = 42.0
    assert matrix.to_numpy_copy()[2][0] == 42.0

    copied_values = matrix.to_numpy_copy()[2]
    copied_values[0] = -5.0
    assert matrix.to_numpy_copy()[2][0] == 42.0

    row_ptr, col_idx, _ = matrix.to_numpy()
    assert row_ptr.base is not None
    assert col_idx.base is not None

    with pytest.raises(ValueError, match="Location.HOST"):
        matrix.to_numpy(rx.Location.DEVICE)


def test_sparse_matrix_host_entry_access() -> None:
    mesh = load_mesh()
    matrix = mesh.sparse_matrix(dtype="float32")
    matrix.reset(2.5, location=rx.Location.ALL)

    row_ptr, col_idx, values = matrix.to_numpy_copy()
    row = int(np.flatnonzero(row_ptr[1:] > row_ptr[:-1])[0])
    offset = int(row_ptr[row])
    col = int(col_idx[offset])

    assert matrix.is_non_zero(row, col)
    assert matrix.value(row, col) == pytest.approx(2.5)

    matrix.set_value(row, col, 7.0)
    assert matrix.value(row, col) == pytest.approx(7.0)


def test_sparse_matrix_from_numpy_copy_owns_memory() -> None:
    row_ptr = np.array([0, 2, 3, 4], dtype=np.int32)
    col_idx = np.array([0, 2, 1, 2], dtype=np.int32)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    matrix = rx.SparseMatrix.from_numpy_copy(
        row_ptr,
        col_idx,
        values,
        shape=(3, 3),
        dtype="float32",
    )

    assert matrix.shape == (3, 3)
    assert matrix.nnz == 4
    out_row_ptr, out_col_idx, out_values = matrix.to_numpy_copy()
    np.testing.assert_array_equal(out_row_ptr, row_ptr)
    np.testing.assert_array_equal(out_col_idx, col_idx)
    np.testing.assert_allclose(out_values, values)

    values[:] = -1.0
    np.testing.assert_allclose(
        matrix.to_numpy_copy()[2],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )


def test_sparse_matrix_dense_multiply() -> None:
    mesh = load_mesh()
    matrix = mesh.sparse_matrix(rx.Op.VV, dtype="float32")
    matrix.reset(1.0, location=rx.Location.ALL)

    rhs_values = np.arange(matrix.cols * 2, dtype=np.float32).reshape(
        matrix.cols,
        2,
    )
    rhs = rx.DenseMatrix(
        matrix.cols,
        2,
        dtype="float32",
        location=rx.Location.ALL,
    )
    rhs.from_numpy_copy(rhs_values, target=rx.Location.ALL)

    result = matrix.multiply(rhs)
    result.move(rx.Location.DEVICE, rx.Location.HOST)
    rx.cuda_stream_synchronize()
    result_values = result.to_numpy_copy(source=rx.Location.HOST)

    row_ptr, col_idx, values = matrix.to_numpy_copy()
    expected = np.zeros((matrix.rows, 2), dtype=np.float32)
    for row in range(matrix.rows):
        for offset in range(row_ptr[row], row_ptr[row + 1]):
            expected[row] += values[offset] * rhs_values[col_idx[offset]]

    assert result.shape == (matrix.rows, 2)
    np.testing.assert_allclose(result_values, expected, rtol=1e-5, atol=1e-5)

    stream = 2
    vector_values = np.arange(matrix.cols, dtype=np.float32).reshape(-1, 1)
    vector = rx.DenseMatrix(
        matrix.cols,
        1,
        dtype="float32",
        location=rx.Location.ALL,
    )
    vector.from_numpy_copy(vector_values, target=rx.Location.ALL, stream=stream)    
    vector_result = matrix.multiply_vector(vector, stream=stream)    
    vector_result.move(rx.Location.DEVICE, rx.Location.HOST)
    rx.cuda_stream_synchronize(stream)
    
    vector_result_values = vector_result.to_numpy_copy(source=rx.Location.HOST)
    expected_vector = np.zeros((matrix.rows, 1), dtype=np.float32)
    for row in range(matrix.rows):
        for offset in range(row_ptr[row], row_ptr[row + 1]):
            expected_vector[row, 0] += (
                values[offset] * vector_values[col_idx[offset], 0]
            )

    assert vector_result.shape == (matrix.rows, 1)
    np.testing.assert_allclose(
        vector_result_values, expected_vector, rtol=1e-5, atol=1e-5
    )
    
    bad_vector = rx.DenseMatrix(
        matrix.cols,
        2,
        dtype="float32",
        location=rx.Location.ALL,
    )
    with pytest.raises(ValueError, match="one column"):
        matrix.multiply_vector(bad_vector)
    with pytest.raises(TypeError):
        matrix.multiply_vector(vector, alpha=1.0)
    with pytest.raises(TypeError):
        matrix.multiply_vector(vector, beta=0.0)


def test_sparse_matrix_torch_csr_cpu_zero_copy() -> None:
    mesh = load_mesh()
    matrix = mesh.sparse_matrix(dtype="float32")
    matrix.reset(1.0, location=rx.Location.ALL)
    torch_matrix = matrix.to_torch(rx.Location.HOST)    
    crow = torch_matrix.crow_indices()
    col = torch_matrix.col_indices()
    values = torch_matrix.values()
    assert torch_matrix.layout == torch.sparse_csr
    assert torch_matrix.crow_indices().data_ptr() == crow.data_ptr()
    assert torch_matrix.col_indices().data_ptr() == col.data_ptr()
    assert torch_matrix.values().data_ptr() == values.data_ptr()
    values[0] = 11.0
    assert matrix.to_numpy_copy()[2][0] == pytest.approx(11.0)


def test_sparse_matrix_from_torch_copy_cpu() -> None:
    crow = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    col = torch.tensor([0, 2, 1, 2], dtype=torch.int32)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    torch_matrix = torch.sparse_csr_tensor(crow, col, values, size=(3, 3))

    matrix = rx.SparseMatrix.from_torch_copy(torch_matrix)
    row_ptr, col_idx, out_values = matrix.to_numpy_copy()
    np.testing.assert_array_equal(row_ptr, crow.numpy())
    np.testing.assert_array_equal(col_idx, col.numpy())
    np.testing.assert_allclose(out_values, values.numpy())

    values[:] = -1.0
    np.testing.assert_allclose(
        matrix.to_numpy_copy()[2],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )


def test_sparse_matrix_torch_csr_cuda_zero_copy() -> None:
    mesh = load_mesh()
    matrix = mesh.sparse_matrix(dtype="float32")
    matrix.reset(3.0, location=rx.Location.ALL)
    torch_matrix = matrix.to_torch(rx.Location.DEVICE)    
    crow = torch_matrix.crow_indices()
    col = torch_matrix.col_indices()
    values = torch_matrix.values()
    assert torch_matrix.layout == torch.sparse_csr
    assert torch_matrix.crow_indices().data_ptr() == crow.data_ptr()
    assert torch_matrix.col_indices().data_ptr() == col.data_ptr()
    assert torch_matrix.values().data_ptr() == values.data_ptr()
    values[0] = 17.0
    torch.cuda.synchronize()
    matrix.sync_device_to_host()
    rx.cuda_stream_synchronize()
    assert matrix.to_numpy_copy()[2][0] == pytest.approx(17.0)


def test_sparse_matrix_from_torch_values_copy_cpu() -> None:
    mesh = load_mesh()
    matrix = mesh.sparse_matrix(dtype="float32")
    values = torch.arange(matrix.nnz, dtype=torch.float32)

    matrix.from_torch_values_copy(values, target=rx.Location.ALL)
    np.testing.assert_allclose(
        matrix.to_numpy_copy()[2],
        values.numpy(),
    )


def test_sparse_matrix_from_torch_copy_cuda() -> None:
    crow = torch.tensor([0, 2, 3, 4], dtype=torch.int32, device="cuda")
    col = torch.tensor([0, 2, 1, 2], dtype=torch.int32, device="cuda")
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
    torch_matrix = torch.sparse_csr_tensor(crow, col, values, size=(3, 3))

    matrix = rx.SparseMatrix.from_torch_copy(torch_matrix)
    row_ptr, col_idx, out_values = matrix.to_numpy_copy()
    np.testing.assert_array_equal(row_ptr, np.array([0, 2, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(col_idx, np.array([0, 2, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(
        out_values,
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )


def test_sparse_matrix_from_torch_values_copy_cuda() -> None:
    mesh = load_mesh()
    matrix = mesh.sparse_matrix(dtype="float32")
    values = torch.arange(matrix.nnz, dtype=torch.float32, device="cuda")

    matrix.from_torch_values_copy(values, target=rx.Location.ALL)
    np.testing.assert_allclose(
        matrix.to_numpy_copy()[2],
        np.arange(matrix.nnz, dtype=np.float32),
    )

def test_sparse_matrix_scipy_multiply_vector(tmp_path: Path) -> None:    
    mesh = load_mesh()
    matrix = mesh.sparse_matrix(rx.Op.VV, dtype="float32")
    matrix.reset(1.0, location=rx.Location.ALL)

    scipy_matrix = matrix.to_scipy_csr()
    assert isinstance(scipy_matrix, sp.csr_matrix)
    assert scipy_matrix.shape == matrix.shape
    assert scipy_matrix.nnz == matrix.nnz

    vector = np.arange(matrix.cols, dtype=np.float32)
    result_matrix = matrix.multiply_vector(vector)
    result_matrix.move(rx.Location.DEVICE, rx.Location.HOST)
    rx.cuda_stream_synchronize()
    result = result_matrix.to_numpy_copy(source=rx.Location.HOST)
    expected = scipy_matrix @ vector.reshape(-1, 1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_diff_sparse_matrix_containers() -> None:    
    mesh = load_mesh()

    jacobian = rx.JacobianSparseMatrix(
        mesh,
        [rx.Op.VV],
        [(1, 1)],
        dtype="float32",
    )
    assert isinstance(jacobian, rx.SparseMatrix)
    assert jacobian.num_terms == 1
    assert jacobian.term_num_rows(0) == jacobian.rows
    assert jacobian.term_rows_range(0) == (0, jacobian.rows)
    assert jacobian.nnz > 0

    hessian = rx.HessianSparseMatrix(
        mesh,
        variable_dim=2,
        extra_nnz_entries=0,
        op=rx.Op.VV,
        dtype="float32",
    )
    assert isinstance(hessian, rx.SparseMatrix)
    assert hessian.variable_dim == 2
    assert hessian.shape == (mesh.num_vertices * 2, mesh.num_vertices * 2)
    assert hessian.nnz > 0
    
    for matrix in (jacobian, hessian):
        assert isinstance(matrix, rx.SparseMatrix)
        assert matrix.rows > 0
        assert matrix.cols > 0
        assert matrix.shape == (matrix.rows, matrix.cols)
        assert matrix.nnz > 0
        assert matrix.dtype == "float32"
        assert matrix.to_numpy_copy()[0].shape[0] == matrix.rows + 1
        
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
