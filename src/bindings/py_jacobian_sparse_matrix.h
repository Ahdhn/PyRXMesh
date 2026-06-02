#pragma once

#include "bindings/py_sparse_matrix.h"

#include "rxmesh/diff/jacobian_sparse_matrix.h"

namespace pyrxmesh_py {

inline rxmesh::BlockShape parse_block_shape(const py::handle& item)
{
    py::sequence shape = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(shape) != 2) {
        throw std::invalid_argument(
            "JacobianSparseMatrix block_shapes entries must be (rows, cols).");
    }
    const int rows = shape[0].cast<int>();
    const int cols = shape[1].cast<int>();
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "JacobianSparseMatrix block shape dimensions must be positive.");
    }
    return rxmesh::BlockShape(rows, cols);
}

inline std::vector<rxmesh::BlockShape> parse_block_shapes(
    py::sequence block_shapes)
{
    std::vector<rxmesh::BlockShape> parsed;
    parsed.reserve(static_cast<size_t>(py::len(block_shapes)));
    for (py::handle item : block_shapes) {
        parsed.push_back(parse_block_shape(item));
    }
    return parsed;
}

template <typename T>
struct PyJacobianSparseMatrix : PySparseMatrixT<T>
{
    using NativeT = rxmesh::JacobianSparseMatrix<T>;

    std::shared_ptr<NativeT> jacobian;

    PyJacobianSparseMatrix(std::shared_ptr<rxmesh::RXMeshStatic> mesh,
                           std::vector<rxmesh::Op>               ops,
                           py::sequence                          block_shapes)
    {
        if (!mesh) {
            throw std::invalid_argument(
                "JacobianSparseMatrix requires a live RXMeshStatic owner.");
        }
        auto parsed_block_shapes = parse_block_shapes(block_shapes);
        if (ops.size() != parsed_block_shapes.size()) {
            throw std::invalid_argument(
                "JacobianSparseMatrix requires one block shape per op.");
        }

        jacobian = std::make_shared<NativeT>(*mesh, ops, parsed_block_shapes);

        this->matrix =
            std::static_pointer_cast<rxmesh::SparseMatrix<T>>(jacobian);
        this->mesh_owner = std::move(mesh);
        this->op         = rxmesh::Op::INVALID;
        this->allocated  = rxmesh::LOCATION_ALL;
        this->released   = false;
    }

    int num_terms() const
    {
        return jacobian->get_num_terms();
    }

    int term_num_rows(int term) const
    {
        if (term < 0 || term >= num_terms()) {
            throw std::out_of_range(
                "JacobianSparseMatrix term index is out of range.");
        }
        return jacobian->get_term_num_rows(term);
    }

    py::tuple term_rows_range(int term) const
    {
        if (term < 0 || term >= num_terms()) {
            throw std::out_of_range(
                "JacobianSparseMatrix term index is out of range.");
        }
        auto range = jacobian->get_term_rows_range(term);
        return py::make_tuple(range.first, range.second);
    }
};

}  // namespace pyrxmesh_py
