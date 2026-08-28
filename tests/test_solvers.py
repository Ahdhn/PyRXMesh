import numpy as np
import pyrxmesh as rx
import pytest


def diagonal_system(dtype: str):
    np_dtype = np.float32 if dtype == "float32" else np.float64
    diagonal = np.array([2.0, 3.0, 4.0, 5.0], dtype=np_dtype)
    row_ptr = np.arange(5, dtype=np.int32)
    col_idx = np.arange(4, dtype=np.int32)
    matrix = rx.SparseMatrix.from_numpy_copy(
        row_ptr,
        col_idx,
        diagonal,
        shape=(4, 4),
        dtype=dtype,
    )

    rhs_values = np.array(
        [[2.0, 4.0], [6.0, 9.0], [8.0, 12.0], [10.0, 15.0]],
        dtype=np_dtype,
    )
    rhs = rx.DenseMatrix(
        4,
        2,
        dtype=dtype,
        location="all",
    )
    rhs.from_numpy_copy(rhs_values, target="all")
    expected = rhs_values / diagonal.reshape(-1, 1)
    return matrix, rhs, expected


def test_cg_solver() -> None:
    matrix, rhs, expected = diagonal_system("float32")
    solver = rx.CGSolver(
        matrix,
        unknown_dim=rhs.cols,
        max_iter=64,
        abs_tol=1e-7,
        rel_tol=0.0,
    )

    solution = solver.solve(rhs)

    assert solver.name == "CG"
    assert solver.iter_taken <= 64
    assert solver.start_residual > 0.0
    assert solver.final_residual < solver.start_residual
    assert solution.shape == expected.shape
    rx.cuda_stream_synchronize()
    np.testing.assert_allclose(
        solution.to_numpy_copy(source="host"),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_pcg_solver() -> None:
    matrix, rhs, expected = diagonal_system("float64")
    solver = rx.PCGSolver(
        matrix,
        unknown_dim=rhs.cols,
        max_iter=64,
        abs_tol=1e-10,
        rel_tol=0.0,
    )
    solution = rx.DenseMatrix(
        matrix.cols,
        rhs.cols,
        dtype="float64",
        location="all",
    )
    solution.reset(0.0, location="all")

    solver.pre_solve(rhs, solution)
    solver.solve(rhs, solution, pre_solve=False)

    assert solver.name == "PCG"
    assert solver.iter_taken <= 64
    assert solver.final_residual < solver.start_residual
    rx.cuda_stream_synchronize()
    np.testing.assert_allclose(
        solution.to_numpy_copy(source="host"),
        expected,
        rtol=1e-8,
        atol=1e-8,
    )


def test_iterative() -> None:
    row_ptr = np.array([0, 1, 2], dtype=np.int32)
    col_idx = np.array([0, 1], dtype=np.int32)
    values = np.array([1, 1], dtype=np.int32)
    int_matrix = rx.SparseMatrix.from_numpy_copy(
        row_ptr,
        col_idx,
        values,
        shape=(2, 2),
        dtype="int32",
    )
    with pytest.raises(ValueError, match="float32 and float64"):
        rx.CGSolver(int_matrix)

    matrix, rhs, _ = diagonal_system("float32")
    solver = rx.CGSolver(matrix, unknown_dim=rhs.cols)
    bad_rhs = rx.DenseMatrix(
        rhs.rows,
        rhs.cols + 1,
        dtype="float32",
        location="all",
    )
    with pytest.raises(ValueError, match="unknown_dim"):
        solver.solve(bad_rhs)


@pytest.mark.parametrize(
    ("solver_cls", "expected_name", "rtol", "atol"),
    [
        (rx.CholeskySolver, "Cholesky", 1e-4, 1e-4),
        (rx.QRSolver, "QR", 1e-4, 1e-4),
        (rx.LUSolver, "LU", 1e-5, 1e-5),
    ],
)
def test_direct_solvers(
    solver_cls,
    expected_name: str,
    rtol: float,
    atol: float,
) -> None:
    matrix, rhs, expected = diagonal_system("float32")
    solver = solver_cls(matrix, permute="none")

    solution = solver.solve(rhs)

    assert solver.name == expected_name
    assert solver.permute == "none"
    assert not solver.is_factorized
    rx.cuda_stream_synchronize()
    np.testing.assert_allclose(
        solution.to_numpy_copy(source="host"),
        expected,
        rtol=rtol,
        atol=atol,
    )


def test_direct_solver_solve() -> None:
    matrix, rhs, expected = diagonal_system("float64")
    solver = rx.LUSolver(matrix, permute="none")
    solution = rx.DenseMatrix(
        matrix.cols,
        rhs.cols,
        dtype="float64",
        location="all",
    )
    solution.reset(0.0, location="all")

    solver.solve(rhs, solution)

    rx.cuda_stream_synchronize()
    np.testing.assert_allclose(
        solution.to_numpy_copy(source="host"),
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_cudss_cholesky_availability() -> None:
    matrix, _, _ = diagonal_system("float32")

    if rx.has_cudss:
        solver = rx.cuDSSCholeskySolver(matrix, permute="none")
        assert solver.name == "cuDSSCholesky"
    else:
        with pytest.raises(RuntimeError, match="PYRXMESH_USE_CUDSS=ON"):
            rx.cuDSSCholeskySolver(matrix, permute="none")

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
