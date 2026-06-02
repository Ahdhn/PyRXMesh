#pragma once

#include "bindings/py_sparse_matrix.h"

#include "rxmesh/diff/hessian_sparse_matrix.h"

namespace pyrxmesh_py {

inline void validate_hessian_inputs(std::shared_ptr<rxmesh::RXMeshStatic> mesh,
                                    int variable_dim,
                                    int extra_nnz_entries)
{
    if (!mesh) {
        throw std::invalid_argument(
            "HessianSparseMatrix requires a live RXMeshStatic owner.");
    }
    if (extra_nnz_entries < 0) {
        throw std::invalid_argument(
            "HessianSparseMatrix extra_nnz_entries must be non-negative.");
    }
    if (variable_dim != 1 && variable_dim != 2 && variable_dim != 3 &&
        variable_dim != 4 && variable_dim != 6) {
        throw std::invalid_argument(
            "HessianSparseMatrix currently supports variable_dim 1, 2, 3, 4, "
            "or 6.");
    }
}

template <typename T, int K>
struct PyHessianSparseMatrix : PySparseMatrixT<T>
{
    using NativeT = rxmesh::HessianSparseMatrix<T, K>;

    std::shared_ptr<NativeT> hessian;

    PyHessianSparseMatrix(std::shared_ptr<rxmesh::RXMeshStatic> mesh,
                          int                                   extra_nnz_entries,
                          rxmesh::Op                            op)
    {
        if (!mesh) {
            throw std::invalid_argument(
                "HessianSparseMatrix requires a live RXMeshStatic owner.");
        }
        if (extra_nnz_entries < 0) {
            throw std::invalid_argument(
                "HessianSparseMatrix extra_nnz_entries must be non-negative.");
        }

        hessian = std::make_shared<NativeT>(*mesh, extra_nnz_entries, op);

        this->matrix =
            std::static_pointer_cast<rxmesh::SparseMatrix<T>>(hessian);
        this->mesh_owner = std::move(mesh);
        this->op         = op;
        this->allocated  = rxmesh::LOCATION_ALL;
        this->released   = false;
    }

    int variable_dim() const
    {
        return K;
    }
};

}  // namespace pyrxmesh_py
