"""NumPy interop helpers attached to ``SparseMatrix``.

Overrides ``SparseMatrix.multiply_vector`` so it accepts a NumPy 1D/2D array
in addition to a ``DenseMatrix``. NumPy is already a hard PyRXMesh
dependency, so this module is always importable.
"""

from __future__ import annotations

import numpy as np

from . import DenseMatrix, SparseMatrix


_spmat_multiply_vector = SparseMatrix.multiply_vector


def _sparse_matrix_multiply_vector(self, vector, stream=None):
    if isinstance(vector, DenseMatrix):
        return _spmat_multiply_vector(self, vector, stream)

    values = np.asarray(vector)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(
            "SparseMatrix.multiply_vector() expects a 1D array or a "
            "DenseMatrix/array with shape (cols, 1)."
        )
    if values.shape[0] != self.cols:
        raise ValueError(
            "SparseMatrix.multiply_vector() vector length must match "
            "matrix.cols."
        )

    dense = DenseMatrix(
        self.cols,
        1,
        dtype=self.dtype,
        location="all",
    )
    dense.from_numpy_copy(values, target="all", stream=stream)
    return _spmat_multiply_vector(self, dense, stream)


SparseMatrix.multiply_vector = _sparse_matrix_multiply_vector
