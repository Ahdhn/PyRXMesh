"""Lazy SciPy helpers attached to ``SparseMatrix``."""

from __future__ import annotations

from typing import Any

from . import SparseMatrix



def _require_scipy_sparse(api: str) -> Any:
    try:
        import scipy.sparse as scipy_sparse
    except ImportError as exc:
        raise ImportError(
            f"SciPy is required for SparseMatrix.{api}(). "
            "Install scipy or use to_numpy()/to_numpy_copy()."
        ) from exc
    return scipy_sparse


def _sparse_matrix_to_scipy_csr(self: SparseMatrix) -> Any:
    scipy_sparse = _require_scipy_sparse("to_scipy_csr")
    row_ptr, col_idx, values = self.to_numpy("host")
    return scipy_sparse.csr_matrix(
        (values, col_idx, row_ptr), shape=self.shape
    )


def _sparse_matrix_to_scipy_csr_copy(
    self: SparseMatrix,
    source: str | int = "host",
    stream: int | None = None,
) -> Any:
    scipy_sparse = _require_scipy_sparse("to_scipy_csr_copy")
    row_ptr, col_idx, values = self.to_numpy_copy(
        source=source, stream=stream
    )
    return scipy_sparse.csr_matrix(
        (values, col_idx, row_ptr), shape=self.shape
    )


SparseMatrix.to_scipy_csr = _sparse_matrix_to_scipy_csr
SparseMatrix.to_scipy_csr_copy = _sparse_matrix_to_scipy_csr_copy
