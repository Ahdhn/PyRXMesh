"""SciPy interop helpers attached to ``SparseMatrix``.

This module is imported automatically by ``pyrxmesh/__init__.py`` and patches
``to_scipy_csr`` / ``to_scipy_csr_copy`` onto ``SparseMatrix``. ``scipy`` is
imported lazily on first call, so importing this module is cheap.
"""

from __future__ import annotations

from . import Location, SparseMatrix


def _require_scipy_sparse(api):
    try:
        import scipy.sparse as scipy_sparse
    except ImportError as exc:
        raise ImportError(
            f"SciPy is required for SparseMatrix.{api}(). "
            "Install scipy or use to_numpy()/to_numpy_copy()."
        ) from exc
    return scipy_sparse


def _sparse_matrix_to_scipy_csr(self):
    scipy_sparse = _require_scipy_sparse("to_scipy_csr")
    row_ptr, col_idx, values = self.to_numpy(Location.HOST)
    return scipy_sparse.csr_matrix(
        (values, col_idx, row_ptr), shape=self.shape
    )


def _sparse_matrix_to_scipy_csr_copy(self, source=Location.HOST, stream=None):
    scipy_sparse = _require_scipy_sparse("to_scipy_csr_copy")
    row_ptr, col_idx, values = self.to_numpy_copy(source=source, stream=stream)
    return scipy_sparse.csr_matrix(
        (values, col_idx, row_ptr), shape=self.shape
    )


SparseMatrix.to_scipy_csr = _sparse_matrix_to_scipy_csr
SparseMatrix.to_scipy_csr_copy = _sparse_matrix_to_scipy_csr_copy
